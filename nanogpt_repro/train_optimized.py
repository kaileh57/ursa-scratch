#!/usr/bin/env python3
"""
Optimized GPT-2 Training Script for RTX A6000/A4500 GPUs

This script demonstrates how to use GPU-specific configurations and optimizations
based on the techniques discussed in the nanoGPT video series.

Usage:
    python train_optimized.py --gpu rtx_a6000
    python train_optimized.py --gpu rtx_a4500
    python train_optimized.py --gpu auto  # auto-detect GPU and use best config
"""

import os
import sys
import argparse
import time
import math
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add configs directory to path
sys.path.append(str(Path(__file__).parent / "configs"))

# Import the base training script components
from train_gpt2 import GPT, GPTConfig, DataLoaderLite

def detect_gpu():
    """Auto-detect GPU type and return appropriate config name"""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    if "a6000" in gpu_name:
        return "rtx_a6000"
    elif "a4500" in gpu_name:
        return "rtx_a4500"
    elif "rtx" in gpu_name or "quadro" in gpu_name:
        # Default to A4500 config for other professional GPUs
        print(f"Detected {gpu_name}, using RTX A4500 config as fallback")
        return "rtx_a4500"
    else:
        print(f"Unknown GPU {gpu_name}, using RTX A4500 config as fallback")
        return "rtx_a4500"

def load_gpu_config(gpu_type):
    """Load GPU-specific configuration"""
    try:
        config_module = importlib.import_module(gpu_type)
        return config_module.get_config()
    except ImportError:
        raise ValueError(f"Config for {gpu_type} not found. Available: rtx_a6000, rtx_a4500")

def setup_optimizations(config):
    """Setup PyTorch optimizations based on config"""
    opts = config['optimizations']
    
    # Enable TF32 for faster training on Ampere GPUs
    if opts.get('tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for ~8x speedup on matrix operations")
    
    # Set memory format for better performance
    torch.backends.cudnn.benchmark = True
    
    # Set float32 matmul precision for memory efficiency
    if opts.get('mixed_precision', True):
        torch.set_float32_matmul_precision('high')
        print("✓ Mixed precision training enabled")
    
    return True

def create_model(config):
    """Create GPT model with optimized config"""
    model_config = GPTConfig(
        block_size=config['model']['block_size'],
        vocab_size=config['model']['vocab_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd']
    )
    
    model = GPT(model_config)
    print(f"✓ Created GPT model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    return model

def setup_optimizer(model, config, device_type):
    """Setup optimized AdamW optimizer"""
    training_config = config['training']
    
    # Configure optimizer with weight decay
    optimizer = model.configure_optimizers(
        weight_decay=training_config['weight_decay'],
        learning_rate=training_config['learning_rate'],
        device_type=device_type,
        master_process=True
    )
    
    print(f"✓ Optimizer configured with lr={training_config['learning_rate']:.2e}")
    return optimizer

def get_lr(step, config):
    """Cosine learning rate schedule with warmup"""
    training_config = config['training']
    max_lr = training_config['learning_rate']
    min_lr = max_lr * training_config['min_lr_ratio']
    warmup_steps = training_config['warmup_steps']
    max_steps = training_config['max_steps']
    
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Cosine decay
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train_step(model, data_loader, optimizer, config, device, step):
    """Optimized training step with mixed precision and gradient accumulation"""
    training_config = config['training']
    opts = config['optimizations']
    
    model.train()
    
    # Calculate dynamic gradient accumulation steps
    total_batch_size = training_config['total_batch_size']
    B = training_config['batch_size']
    T = training_config['sequence_length']
    grad_accum_steps = total_batch_size // (B * T)
    
    optimizer.zero_grad()
    loss_accum = 0.0
    
    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # Mixed precision forward pass
        if opts.get('mixed_precision', True):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        # Backward pass
        loss.backward()
    
    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip'])
    
    # Learning rate scheduling
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Optimizer step
    optimizer.step()
    
    return loss_accum.item(), norm.item(), lr

def main():
    parser = argparse.ArgumentParser(description='Optimized GPT-2 Training')
    parser.add_argument('--gpu', choices=['rtx_a6000', 'rtx_a4500', 'auto'], 
                       default='auto', help='GPU configuration to use')
    parser.add_argument('--steps', type=int, default=100, 
                       help='Number of training steps')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    # Setup device
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        return
    
    device = "cuda"
    device_type = "cuda"
    
    # Detect or load GPU configuration  
    if args.gpu == 'auto':
        gpu_type = detect_gpu()
        if gpu_type is None:
            print("❌ Could not detect compatible GPU")
            return
    else:
        gpu_type = args.gpu
    
    print(f"🚀 Using {gpu_type} configuration")
    
    # Load configuration
    config = load_gpu_config(gpu_type)
    
    # Setup optimizations
    setup_optimizations(config)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Compile model for faster training
    if config['optimizations'].get('compile', True):
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model)
        print("✓ Model compilation complete")
    
    # Setup optimizer
    optimizer = setup_optimizer(model, config, device_type)
    
    # Create data loader
    B = config['training']['batch_size']
    T = config['training']['sequence_length']
    data_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train", data_dir="/mnt/raid0/edu_fineweb10B", master_process=True)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="val", data_dir="/mnt/raid0/edu_fineweb10B", master_process=True)

    print("\n🏋️  Starting optimized training...")
    print(f"📊 Batch size: {B}, Sequence length: {T}")
    print(f"🔢 Total batch size: {config['training']['total_batch_size']:,} tokens")
    
    # Training loop with performance monitoring
    start_time = time.time()
    
    for step in range(args.steps):
        step_start = time.time()
        
        # Training step
        loss, grad_norm, lr = train_step(model, data_loader, optimizer, config, device, step)
        
        # Synchronize GPU for accurate timing
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        # Calculate throughput
        total_batch_size = config['training']['total_batch_size']
        tokens_per_sec = total_batch_size / step_time
        
        # Print progress
        if step % 10 == 0 or step == args.steps - 1:
            print(f"Step {step:4d} | Loss: {loss:.6f} | LR: {lr:.4e} | "
                  f"Grad Norm: {grad_norm:.4f} | Time: {step_time*1000:.1f}ms | "
                  f"Tokens/sec: {tokens_per_sec:,.0f}")
    
    total_time = time.time() - start_time
    avg_tokens_per_sec = (config['training']['total_batch_size'] * args.steps) / total_time
    
    print(f"\n✅ Training complete!")
    print(f"📈 Average throughput: {avg_tokens_per_sec:,.0f} tokens/sec")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Save the model checkpoint
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, "model_optimized.pt")
    
    # unwrap model if compiled
    unwrapped_model = model._orig_mod if config['optimizations'].get('compile', True) else model
    
    checkpoint = {
        'model': unwrapped_model.state_dict(),
        'config': unwrapped_model.config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Model saved to {checkpoint_path}")
    
    # Performance summary
    print(f"\n📋 Performance Summary for {gpu_type.upper()}:")
    print(f"   • Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"   • Memory used: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print(f"   • Peak throughput: {avg_tokens_per_sec:,.0f} tokens/sec")
    
    if args.profile:
        print(f"   • GPU utilization: {torch.cuda.utilization()}%")

if __name__ == "__main__":
    main()