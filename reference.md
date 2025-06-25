/*
This file is not intended to be #included, but instead read and parsed by the build system.
The C build system (e.g. Makefile) will concatenate all the .c files into a single .cu file,
and this will be compiled by nvcc. This is done to speed up compilation, which is otherwise
very slow on a large number of small .c files.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cuda_common.h"
#include "cublas_common.h"
#include "attention.h"
#include "fused_classifier.h"
#include "layernorm.h"
#include "gelu.h"
#include "adamw.h"
#include "schedulers.h"
#include "dataloader.h"
#include "utils.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte that corresponds to the token
            const float* wte_ix = wte + ix * C;
            // seek to the position in wpe that corresponds to the position
            const float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float* dwte, float* dwpe, const float* dout, const int* inp, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                dwte_ix[i] += dout_bt[i];
                dwpe_t[i] += dout_bt[i];
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd, const float* inp, const float* weight, const float* bias, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m /= C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v /= C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + 1e-5f);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (x[i] - m) * s; // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o;
            }
            // cache the mean and rstd for the backward pass
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            const float* weight_bt = weight; // note: weight is not indexed by b,t
            float* dinp_bt = dinp + b * T * C + t * C;
            float* dweight_bt = dweight; // note: dweight is not indexed by b,t
            float* dbias_bt = dbias; // note: dbias is not indexed by b,t
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = dout_bt[i] * weight_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C;
            dnorm_norm_mean /= C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = dout_bt[i] * weight_bt[i];
                // gradient for inp
                float dinp_i = 0.0f;
                dinp_i += dnorm_i; // term 1
                dinp_i -= dnorm_mean; // term 2
                dinp_i -= norm_bti * dnorm_norm_mean; // term 3
                dinp_i *= rstd_bt; // final scale
                dinp_bt[i] += dinp_i;
                // gradient for weight
                dweight_bt[i] += dout_bt[i] * norm_bti;
                // gradient for bias
                dbias_bt[i] += dout_bt[i];
            }
        }
    }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, int B, int T, int C, int OC) {
    // inp is (B, T, C), weight is (OC, C), bias is (OC)
    // out will be (B, T, OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, int B, int T, int C, int OC) {
    // inp (B,T,C), weight (OC,C), bias (OC), dout (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o * C;
                float d = dout_bt[o];
                // backward to bias
                if (dbias != NULL) {
                    dbias[o] += d;
                }
                // backward to weight
                float* dwrow = dweight + o * C;
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp[b * T * C + t * C + i] * d;
                }
                // backward to input
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
}


void attention_forward(float* out, float* preatt, float* att, const float* inp, const float* weight, const float* bias, int B, int T, int C, int NH) {
    // inp is (B, T, 3C) holding the concatenated q,k,v projections
    // out is (B, T, C)
    // preatt, att are buffers, sized (B, NH, T, T)
    int C3 = C*3;
    int HS = C / NH; // head size

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * HS;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * HS + C; // +C because key is at offset C
                    // (q, k) dot product
                    float val = 0.0f;
                    for (int i = 0; i < HS; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val /= sqrtf(HS);
                    preatt_bth[t2] = val;
                }
            }
        }
    }

    // softmax
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                float* att_btht = att + b * NH * T * T + h * T * T + t * T;
                const float* preatt_btht = preatt + b * NH * T * T + h * T * T + t * T;

                // find max value for numerical stability
                float max_val = preatt_btht[0];
                for (int t2 = 1; t2 <= t; t2++) {
                    if (preatt_btht[t2] > max_val) {
                        max_val = preatt_btht[t2];
                    }
                }

                // exp and sum
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float exp_val = expf(preatt_btht[t2] - max_val);
                    att_btht[t2] = exp_val;
                    sum += exp_val;
                }
                float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

                // normalize
                for (int t2 = 0; t2 <= t; t2++) {
                    att_btht[t2] *= inv_sum;
                }
            }
        }
    }

    // weighted sum of values
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bth = out + b * T * C + t * C;
            for (int h = 0; h < NH; h++) {
                const float* att_btht = att + b * NH * T * T + h * T * T + t * T;
                float* out_bth_h = out_bth + h * HS;
                for (int i = 0; i < HS; i++) {
                    out_bth_h[i] = 0.0f;
                }

                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * HS + 2*C; // +2*C because value is at offset 2*C
                    float att_val = att_btht[t2];
                    for (int i = 0; i < HS; i++) {
                        out_bth_h[i] += att_val * value_t2[i];
                    }
                }
            }
        }
    }
}


void attention_backward(float* dinp, float* dpreatt, float* datt, const float* dout, const float* inp, const float* att, int B, int T, int C, int NH) {
    // inp (B, T, 3C)
    // att (B, NH, T, T)
    // dout (B, T, C)
    // dinp (B, T, 3C)
    int C3 = C * 3;
    int HS = C / NH;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* att_btht = att + b * NH * T * T + h * T * T + t * T;
                const float* dout_bth = dout + b * T * C + t * C + h * HS;

                // backward pass through weighted sum of values
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * HS + 2*C;
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * HS + 2*C;
                    float* datt_btht = datt + b * NH * T * T + h * T * T + t * T;
                    for (int i = 0; i < HS; i++) {
                        datt_btht[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_btht[t2] * dout_bth[i];
                    }
                }
            }
        }
    }

    // backward pass through softmax
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float* att_btht = att + b * NH * T * T + h * T * T + t * T;
                const float* datt_btht = datt + b * NH * T * T + h * T * T + t * T;
                float* dpreatt_btht = dpreatt + b * NH * T * T + h * T * T + t * T;

                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float tmp = (t2 == t3) - att_btht[t2];
                        dpreatt_btht[t3] += att_btht[t3] * datt_btht[t2] * tmp;
                    }
                }
            }
        }
    }

    // backward pass through dot product
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * HS;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * HS;
                float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;

                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * HS + C;
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * HS + C;

                    float d_preatt = dpreatt_bth[t2] / sqrtf(HS);
                    for (int i = 0; i < HS; i++) {
                        dquery_t[i] += key_t2[i] * d_preatt;
                        dkey_t2[i] += query_t[i] * d_preatt;
                    }
                }
            }
        }
    }
}


void gelu_forward(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

void gelu_backward(float* dinp, const float* inp, const float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float x3 = x * x * x;
        float c = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        float tanh_c = tanhf(c);
        float sech_c = 1.0f - tanh_c * tanh_c;
        float dc_dx = sqrtf(2.0f / M_PI) * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += (0.5f * (1.0f + tanh_c) + 0.5f * x * sech_c * dc_dx) * dout[i];
    }
}

void residual_forward(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, const float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, const float* logits, int B, int T, int V) {
    // output: probs is (B,T,V) of the probabilities
    // input: logits is (B,T,V) of the unnormalized log probabilities
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // offset to the current row
            const float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // find the max value in this row for stability
            float max_val = logits_bt[0];
            for (int i = 1; i < V; i++) {
                if (logits_bt[i] > max_val) {
                    max_val = logits_bt[i];
                }
            }
            // exp and sum
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - max_val);
                sum += probs_bt[i];
            }
            // normalize
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

void crossentropy_forward(float* losses, const float* probs, const int* targets, int B, int T) {
    // output: losses is a single float
    // input: probs are the probabilities (B,T,V)
    // input: targets are the indices of the correct classes (B,T)
    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss for this time step
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            loss += -logf(probs_bt[ix]);
        }
    }
    *losses = loss / (B * T);
}

void crossentropy_backward(float* dlogits, const float* probs, const int* targets, int B, int T, int V) {
    // backwards through log and then softmax
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            // derivative of loss with respect to logits is probs - 1(at target)
            for(int i=0; i<V; ++i) {
                float indicator = (i == ix) ? 1.0f : 0.0f;
                dlogits_bt[i] += (probs_bt[i] - indicator) / (B*T);
            }
        }
    }
}


void adamw_optimizer(float* params, float* grads, float* m, float* v, long long num_params, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    for(long long i=0; i<num_params; ++i) {
        float grad = grads[i];
        // weight decay
        params[i] -= learning_rate * params[i] * weight_decay;
        // update momentum
        m[i] = beta1 * m[i] + (1 - beta1) * grad;
        // update velocity
        v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
        // bias correction
        float m_hat = m[i] / (1 - powf(beta1, t));
        float v_hat = v[i] / (1 - powf(beta2, t));
        // update params
        params[i] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length
    int vocab_size; // vocab size
    int num_layers; // number of layers
    int num_heads; // number of heads
    int channels; // number of channels
} GPT2Config;

typedef struct {
    // weights
    float* wte; // (V, C)
    float* wpe; // (T, C)
    float* ln1w, *ln1b; // (L, C)
    float* c_attn_w, *c_attn_b; // (L, 3C, C)
    float* c_proj_w, *c_proj_b; // (L, C, C)
    float* ln2w, *ln2b; // (L, C)
    float* c_fc_w, *c_fc_b; // (L, 4C, C)
    float* c_proj2_w, *c_proj2_b; // (L, C, 4C)
    float* lnfw, *lnfb; // (C)
    // gradients
    float* dwte;
    float* dwpe;
    float* dln1w, *dln1b;
    float* dc_attn_w, *dc_attn_b;
    float* dc_proj_w, *dc_proj_b;
    float* dln2w, *dln2b;
    float* dc_fc_w, *dc_fc_b;
    float* dc_proj2_w, *dc_proj2_b;
    float* dlnfw, *dlnfb;
    // model settings
    GPT2Config config;
    int num_parameters;
} GPT2;

void gpt2_build_from_checkpoint(GPT2* model, const char* checkpoint_path) {
    // read in the config file
    FILE *f_cfg = fopen("gpt2_124M_config.bin", "rb");
    if (f_cfg == NULL) { printf("Error opening config file\n"); exit(1); }
    fread(&model->config, sizeof(GPT2Config), 1, f_cfg);
    fclose(f_cfg);

    // calculate the number of parameters
    int V = model->config.vocab_size;
    int T = model->config.max_seq_len;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    model->num_parameters = 2*L*C + L*3*C*C + L*C*C + 2*L*C + L*4*C*C + L*C*4*C + 2*C;
    printf("Number of parameters: %d\n", model->num_parameters);

    // allocate memory for the weights and gradients
    model->wte = (float*)malloc(V * C * sizeof(float));
    model->wpe = (float*)malloc(T * C * sizeof(float));
    model->ln1w = (float*)malloc(L * C * sizeof(float));
    model->ln1b = (float*)malloc(L * C * sizeof(float));
    model->c_attn_w = (float*)malloc(L * 3*C * C * sizeof(float));
    model->c_attn_b = (float*)malloc(L * 3*C * sizeof(float));
    model->c_proj_w = (float*)malloc(L * C * C * sizeof(float));
    model->c_proj_b = (float*)malloc(L * C * sizeof(float));
    model->ln2w = (float*)malloc(L * C * sizeof(float));
    model->ln2b = (float*)malloc(L * C * sizeof(float));
    model->c_fc_w = (float*)malloc(L * 4*C * C * sizeof(float));
    model->c_fc_b = (float*)malloc(L * 4*C * sizeof(float));
    model->c_proj2_w = (float*)malloc(L * C * 4*C * sizeof(float));
    model->c_proj2_b = (float*)malloc(L * C * sizeof(float));
    model->lnfw = (float*)malloc(C * sizeof(float));
    model->lnfb = (float*)malloc(C * sizeof(float));

    // read in the weights from the checkpoint file
    FILE *f_w = fopen(checkpoint_path, "rb");
    if (f_w == NULL) { printf("Error opening weights file\n"); exit(1); }
    fread(model->wte, sizeof(float), V * C, f_w);
    fread(model->wpe, sizeof(float), T * C, f_w);
    fread(model->ln1w, sizeof(float), L * C, f_w);
    fread(model->ln1b, sizeof(float), L * C, f_w);
    fread(model->c_attn_w, sizeof(float), L * 3*C * C, f_w);
    fread(model->c_attn_b, sizeof(float), L * 3*C, f_w);
    fread(model->c_proj_w, sizeof(float), L * C * C, f_w);
    fread(model->c_proj_b, sizeof(float), L * C, f_w);
    fread(model->ln2w, sizeof(float), L * C, f_w);
    fread(model->ln2b, sizeof(float), L * C, f_w);
    fread(model->c_fc_w, sizeof(float), L * 4*C * C, f_w);
    fread(model->c_fc_b, sizeof(float), L * 4*C, f_w);
    fread(model->c_proj2_w, sizeof(float), L * C * 4*C, f_w);
    fread(model->c_proj2_b, sizeof(float), L * C, f_w);
    fread(model->lnfw, sizeof(float), C, f_w);
    fread(model->lnfb, sizeof(float), C, f_w);
    fclose(f_w);

    // allocate memory for the gradients
    model->dwte = (float*)calloc(V * C, sizeof(float));
    model->dwpe = (float*)calloc(T * C, sizeof(float));
    model->dln1w = (float*)calloc(L * C, sizeof(float));
    model->dln1b = (float*)calloc(L * C, sizeof(float));
    model->dc_attn_w = (float*)calloc(L * 3*C * C, sizeof(float));
    model->dc_attn_b = (float*)calloc(L * 3*C, sizeof(float));
    model->dc_proj_w = (float*)calloc(L * C * C, sizeof(float));
    model->dc_proj_b = (float*)calloc(L * C, sizeof(float));
    model->dln2w = (float*)calloc(L * C, sizeof(float));
    model->dln2b = (float*)calloc(L * C, sizeof(float));
    model->dc_fc_w = (float*)calloc(L * 4*C * C, sizeof(float));
    model->dc_fc_b = (float*)calloc(L * 4*C, sizeof(float));
    model->dc_proj2_w = (float*)calloc(L * C * 4*C, sizeof(float));
    model->dc_proj2_b = (float*)calloc(L * C, sizeof(float));
    model->dlnfw = (float*)calloc(C, sizeof(float));
    model->dlnfb = (float*)calloc(C, sizeof(float));
}

void gpt2_forward(GPT2 *model, int* inp, int* targets, int B, int T) {
    // a giant ((B,T) + (B,T)) -> B*T*V array of numbers
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // forward pass
    float* x = (float*)malloc(B * T * C * sizeof(float));
    encoder_forward(x, inp, model->wte, model->wpe, B, T, C);

    float* ln1_mean = (float*)malloc(B * T * sizeof(float));
    float* ln1_rstd = (float*)malloc(B * T * sizeof(float));
    float* c_attn_out = (float*)malloc(B * T * 3*C * sizeof(float));
    float* att_preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att_att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* c_proj_out = (float*)malloc(B * T * C * sizeof(float));
    float* residual2 = (float*)malloc(B * T * C * sizeof(float));
    float* ln2_mean = (float*)malloc(B * T * sizeof(float));
    float* ln2_rstd = (float*)malloc(B * T * sizeof(float));
    float* c_fc_out = (float*)malloc(B * T * 4*C * sizeof(float));
    float* gelu_out = (float*)malloc(B * T * 4*C * sizeof(float));
    float* c_proj2_out = (float*)malloc(B * T * C * sizeof(float));
    float* residual3 = (float*)malloc(B * T * C * sizeof(float));

    for (int l = 0; l < L; l++) {
        float* x_res = x;
        // layer norm 1
        layernorm_forward(x, ln1_mean, ln1_rstd, x_res, model->ln1w + l*C, model->ln1b + l*C, B, T, C);
        // attention
        matmul_forward(c_attn_out, x, model->c_attn_w + l*3*C*C, model->c_attn_b + l*3*C, B, T, C, 3*C);
        attention_forward(c_proj_out, att_preatt, att_att, c_attn_out, model->c_proj_w + l*C*C, model->c_proj_b + l*C, B, T, C, NH);
        // residual 1
        residual_forward(residual2, x_res, c_proj_out, B*T*C);
        // layer norm 2
        layernorm_forward(x, ln2_mean, ln2_rstd, residual2, model->ln2w + l*C, model->ln2b + l*C, B, T, C);
        // MLP
        matmul_forward(c_fc_out, x, model->c_fc_w + l*4*C*C, model->c_fc_b + l*4*C, B, T, C, 4*C);
        gelu_forward(gelu_out, c_fc_out, B*T*4*C);
        matmul_forward(c_proj2_out, gelu_out, model->c_proj2_w + l*C*4*C, model->c_proj2_b + l*C, B, T, 4*C, C);
        // residual 2
        residual_forward(residual3, residual2, c_proj2_out, B*T*C);
        x = residual3;
    }

    float* lnf_mean = (float*)malloc(B * T * sizeof(float));
    float* lnf_rstd = (float*)malloc(B * T * sizeof(float));
    layernorm_forward(x, lnf_mean, lnf_rstd, x, model->lnfw, model->lnfb, B, T, C);

    float* logits = (float*)malloc(B * T * V * sizeof(float));
    matmul_forward(logits, x, model->wte, NULL, B, T, C, V);
    float* probs = (float*)malloc(B * T * V * sizeof(float));
    softmax_forward(probs, logits, B, T, V);

    float loss;
    crossentropy_forward(&loss, probs, targets, B, T);
    printf("Loss: %f\n", loss);


    // free memory
    free(x);
    free(ln1_mean);
    free(ln1_rstd);
    free(c_attn_out);
    free(att_preatt);
    free(att_att);
    free(c_proj_out);
    free(residual2);
    free(ln2_mean);
    free(ln2_rstd);
    free(c_fc_out);
    free(gelu_out);
    free(c_proj2_out);
    free(residual3);
    free(lnf_mean);
    free(lnf_rstd);
    free(logits);
    free(probs);
}

void gpt2_zero_grad(GPT2 *model) {
    memset(model->dwte, 0, model->config.vocab_size * model->config.channels * sizeof(float));
    memset(model->dwpe, 0, model->config.max_seq_len * model->config.channels * sizeof(float));
    memset(model->dln1w, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dln1b, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dc_attn_w, 0, model->config.num_layers * 3*model->config.channels * model->config.channels * sizeof(float));
    memset(model->dc_attn_b, 0, model->config.num_layers * 3*model->config.channels * sizeof(float));
    memset(model->dc_proj_w, 0, model->config.num_layers * model->config.channels * model->config.channels * sizeof(float));
    memset(model->dc_proj_b, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dln2w, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dln2b, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dc_fc_w, 0, model->config.num_layers * 4*model->config.channels * model->config.channels * sizeof(float));
    memset(model->dc_fc_b, 0, model->config.num_layers * 4*model->config.channels * sizeof(float));
    memset(model->dc_proj2_w, 0, model->config.num_layers * model->config.channels * 4*model->config.channels * sizeof(float));
    memset(model->dc_proj2_b, 0, model->config.num_layers * model->config.channels * sizeof(float));
    memset(model->dlnfw, 0, model->config.channels * sizeof(float));
    memset(model->dlnfb, 0, model->config.channels * sizeof(float));
}

void gpt2_update(GPT2* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    int num_params = model->num_parameters;
    float* params[] = {model->wte, model->wpe, model->ln1w, model->ln1b, model->c_attn_w, model->c_attn_b, model->c_proj_w, model->c_proj_b, model->ln2w, model->ln2b, model->c_fc_w, model->c_fc_b, model->c_proj2_w, model->c_proj2_b, model->lnfw, model->lnfb};
    float* grads[] = {model->dwte, model->dwpe, model->dln1w, model->dln1b, model->dc_attn_w, model->dc_attn_b, model->dc_proj_w, model->dc_proj_b, model->dln2w, model->dln2b, model->dc_fc_w, model->dc_fc_b, model->dc_proj2_w, model->dc_proj2_b, model->dlnfw, model->dlnfb};
    long long num_elements[] = {model->config.vocab_size * model->config.channels, model->config.max_seq_len * model->config.channels, model->config.num_layers * model->config.channels, model->config.num_layers * model->config.channels, model->config.num_layers * 3*model->config.channels * model->config.channels, model->config.num_layers * 3*model->config.channels, model->config.num_layers * model->config.channels * model->config.channels, model->config.num_layers * model->config.channels, model->config.num_layers * model->config.channels, model->config.num_layers * model->config.channels, model->config.num_layers * 4*model->config.channels * model->config.channels, model->config.num_layers * 4*model->config.channels, model->config.num_layers * model->config.channels * 4*model->config.channels, model->config.num_layers * model->config.channels, model->config.channels, model->config.channels};

    float* m[16];
    float* v[16];
    for(int i=0; i<16; ++i) {
        m[i] = (float*)calloc(num_elements[i], sizeof(float));
        v[i] = (float*)calloc(num_elements[i], sizeof(float));
    }

    for (int i=0; i<16; ++i) {
        adamw_optimizer(params[i], grads[i], m[i], v[i], num_elements[i], learning_rate, beta1, beta2, eps, weight_decay, t);
    }

    for(int i=0; i<16; ++i) {
        free(m[i]);
        free(v[i]);
    }
}

int main() {
    // build the model
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders
    const char* tiny_stories_train = "data/TinyStories_train.bin";
    const char* tiny_stories_val = "data/TinyStories_val.bin";
    int B = 4;
    int T = 64;
    DataLoader train_loader;
    dataloader_init(&train_loader, tiny_stories_train, B, T);
    printf("train loader num_batches: %d\n", train_loader.num_batches);
    DataLoader val_loader;
    dataloader_init(&val_loader, tiny_stories_val, B, T);
    printf("val loader num_batches: %d\n", val_loader.num_batches);

    // build the tokenizer
    Tokenizer tokenizer;
    tokenizer_build(&tokenizer, "gpt2_tokenizer.bin");

    // training loop
    for (int step = 0; step < 50; step++) {
        // once in a while estimate the loss on validation set
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < 10; i++) {
                dataloader_next_batch(&val_loader);
                int* val_inputs = val_loader.inputs;
                int* val_targets = val_loader.targets;
                gpt2_forward(&model, val_inputs, val_targets, B, T);
                // val_loss += ... TODO
            }
            // printf("Val loss: %f\n", val_loss / 10);
        }

        // training step
        dataloader_next_batch(&train_loader);
        int* inputs = train_loader.inputs;
        int* targets = train_loader.targets;
        gpt2_forward(&model, inputs, targets, B, T);
        gpt2_zero_grad(&model);
        // gpt2_backward(&model); // TODO
        gpt2_update(&model, 1e-4, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    }


    // free everything
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(model.wte);
    free(model.wpe);
    free(model.ln1w);
    free(model.ln1b);
    free(model.c_attn_w);
    free(model.c_attn_b);
    free(model.c_proj_w);
    free(model.c_proj_b);
    free(model.ln2w);
    free(model.ln2b);
    free(model.c_fc_w);
    free(model.c_fc_b);
    free(model.c_proj2_w);
    free(model.c_proj2_b);
    free(model.lnfw);
    free(model.lnfb);
    free(model.dwte);
    free(model.dwpe);
    free(model.dln1w);
    free(model.dln1b);
    free(model.dc_attn_w);
    free(model.dc_attn_b);
    free(model.dc_proj_w);
    free(model.dc_proj_b);
    free(model.dln2w);
    free(model.dln2b);
    free(model.dc_fc_w);
    free(model.dc_fc_b);
    free(model.dc_proj2_w);
    free(model.dc_proj2_b);
    free(model.dlnfw);
    free(model.dlnfb);

    return 0;
}
