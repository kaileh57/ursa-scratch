# RTX A4500 Optimal Configuration  
# RTX A4500: 20GB VRAM, 7,168 CUDA cores, 224 Tensor cores
# 200W TDP, 640 GB/s memory bandwidth

# Model Configuration
BLOCK_SIZE = 1024  # sequence length
VOCAB_SIZE = 50304  # rounded up from 50257 for better GPU efficiency (powers of 2)
N_LAYER = 12      # number of transformer blocks
N_HEAD = 12       # number of attention heads
N_EMBD = 768      # embedding dimension

# Training Configuration  
BATCH_SIZE = 16           # smaller batch size due to 20GB VRAM limit
SEQUENCE_LENGTH = 1024    # full GPT-2 sequence length
TOTAL_BATCH_SIZE = 524288 # 2**19, ~0.5M tokens
GRAD_ACCUM_STEPS = None   # calculated dynamically

# Optimization Settings
LEARNING_RATE = 6e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Training Schedule
WARMUP_STEPS = 715
MAX_STEPS = 19073

# Hardware Optimizations
USE_MIXED_PRECISION = True    # bfloat16 for Ampere architecture
USE_TF32 = True              # 8x speedup for matrix ops
USE_FLASH_ATTENTION = True   # memory efficient attention - critical for 20GB
USE_COMPILE = True           # torch.compile for kernel fusion
USE_FUSED_OPTIMIZER = True   # fused AdamW when available

# Multi-GPU Settings (if using multiple A4500s)
USE_DDP = True               # distributed data parallel
DDP_BACKEND = 'nccl'
USE_NVLINK = True           # A4500 supports NVLink for multi-GPU

# Memory Management
PIN_MEMORY = True
NON_BLOCKING = True
GRADIENT_CHECKPOINTING = False  # Can enable if hitting memory limits

# Power Efficiency Settings
POWER_LIMIT = 200  # watts - matches A4500 TDP

def get_config():
    """Returns optimized config for RTX A4500"""
    return {
        'model': {
            'block_size': BLOCK_SIZE,
            'vocab_size': VOCAB_SIZE,
            'n_layer': N_LAYER,
            'n_head': N_HEAD,
            'n_embd': N_EMBD,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'total_batch_size': TOTAL_BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'min_lr_ratio': MIN_LR_RATIO,
            'weight_decay': WEIGHT_DECAY,
            'beta1': BETA1,
            'beta2': BETA2,
            'grad_clip': GRAD_CLIP,
            'warmup_steps': WARMUP_STEPS,
            'max_steps': MAX_STEPS,
        },
        'optimizations': {
            'mixed_precision': USE_MIXED_PRECISION,
            'tf32': USE_TF32,
            'flash_attention': USE_FLASH_ATTENTION,
            'compile': USE_COMPILE,
            'fused_optimizer': USE_FUSED_OPTIMIZER,
            'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        },
        'hardware': {
            'ddp': USE_DDP,
            'ddp_backend': DDP_BACKEND,
            'nvlink': USE_NVLINK,
            'pin_memory': PIN_MEMORY,
            'non_blocking': NON_BLOCKING,
            'power_limit': POWER_LIMIT,
        }
    }

# Performance estimates for RTX A4500:
# - Expected training throughput: ~150k+ tokens/sec with all optimizations
# - Memory utilization: ~18-19GB out of 20GB for this configuration
# - Power consumption: ~180-200W under full load
# - Multi-GPU scaling: Near-linear with NVLink between 2 A4500s
# - Note: More memory-constrained than A6000, may need gradient checkpointing for larger models 