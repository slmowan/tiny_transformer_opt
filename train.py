"""
Main training script for comparing optimization algorithms
Supports training with SGD, Momentum, Adagrad, and Adam
"""

import torch
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

from config import Config
from model import GPT, GPTConfig
from optimizers import SGD, MomentumSGD, Adagrad, Adam
from utils import (
    load_data, get_batch, LRScheduler, clip_gradients,
    compute_gradient_norm, evaluate, save_checkpoint,
    generate_sample, MetricsLogger
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, optimizer_name, config):
    """
    Create optimizer based on name
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('sgd', 'momentum', 'adagrad', 'adam')
        config: Configuration object
        
    Returns:
        optimizer: Optimizer instance
    """
    optimizer_configs = config.get_optimizer_configs()
    
    if optimizer_name not in optimizer_configs:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    opt_config = optimizer_configs[optimizer_name]
    
    if optimizer_name == 'sgd':
        return SGD(model.parameters(), **opt_config)
    elif optimizer_name == 'momentum':
        return MomentumSGD(model.parameters(), **opt_config)
    elif optimizer_name == 'adagrad':
        return Adagrad(model.parameters(), **opt_config)
    elif optimizer_name == 'adam':
        return Adam(model.parameters(), **opt_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train(model, train_dataset, val_dataset, optimizer, config, experiment_name):
    """
    Train the model with a specific optimizer
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizer: Optimizer to use
        config: Configuration object
        experiment_name: Name for this experiment
    """
    device = config.device
    model = model.to(device)
    
    # Initialize learning rate scheduler
    if config.use_lr_schedule:
        scheduler = LRScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            decay_steps=config.lr_decay_steps,
            max_lr=optimizer.param_groups[0]['lr'],
            min_lr=config.min_lr
        )
    
    # Initialize logger
    logger = MetricsLogger(config.log_dir, experiment_name)
    
    # Training loop
    model.train()
    step = 0
    total_steps = config.num_epochs * (len(train_dataset) // config.batch_size)
    
    print(f"\n{'='*60}")
    print(f"Training with {experiment_name}")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    with tqdm(total=total_steps, desc=f"Training {experiment_name}") as pbar:
        for epoch in range(config.num_epochs):
            # Shuffle data each epoch by generating random batches
            num_batches = len(train_dataset) // config.batch_size
            
            for batch_idx in range(num_batches):
                # Get batch
                x, y = get_batch(train_dataset, config.batch_size, device)
                
                # Forward pass
                logits, loss = model(x, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm before clipping
                grad_norm = compute_gradient_norm(model)
                
                # Gradient clipping
                if config.use_grad_clip:
                    clip_gradients(model, config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                
                # Update learning rate
                if config.use_lr_schedule:
                    current_lr = scheduler.step()
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                step += 1
                
                # Logging
                if step % config.log_interval == 0:
                    logger.log(
                        step=step,
                        train_loss=loss.item(),
                        learning_rate=current_lr,
                        gradient_norm=grad_norm
                    )
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.6f}',
                        'grad': f'{grad_norm:.4f}'
                    })
                
                # Evaluation
                if step % config.eval_interval == 0:
                    val_loss, val_perplexity = evaluate(
                        model, val_dataset, config.batch_size, device
                    )
                    
                    logger.log(
                        step=step,
                        val_loss=val_loss,
                        val_perplexity=val_perplexity
                    )
                    
                    print(f"\nStep {step} | Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
                    
                    # Generate sample
                    if step % (config.eval_interval * 5) == 0:
                        sample = generate_sample(
                            model, train_dataset, 
                            prompt="ROMEO:", 
                            max_tokens=100, 
                            device=device
                        )
                        print(f"\nGenerated sample:\n{sample}\n")
                
                # Save checkpoint
                if step % config.save_interval == 0:
                    checkpoint_path = os.path.join(
                        config.checkpoint_dir,
                        f"{experiment_name}_step_{step}.pt"
                    )
                    save_checkpoint(model, optimizer, step, loss.item(), checkpoint_path)
                
                pbar.update(1)
    
    # Final evaluation
    print("\nFinal evaluation...")
    val_loss, val_perplexity = evaluate(model, val_dataset, config.batch_size, device)
    logger.log(
        step=step,
        val_loss=val_loss,
        val_perplexity=val_perplexity
    )
    
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Perplexity: {val_perplexity:.2f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{experiment_name}_final.pt"
    )
    save_checkpoint(model, optimizer, step, loss.item(), final_checkpoint_path)
    
    # Save metrics and plot
    logger.save()
    plot_path = os.path.join(config.log_dir, f"{experiment_name}_metrics.png")
    logger.plot_metrics(save_path=plot_path)
    
    print(f"\nTraining complete for {experiment_name}!")
    print(f"Logs saved to {config.log_dir}")
    print(f"Checkpoints saved to {config.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Tiny Transformer LM')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'momentum', 'adagrad', 'adam'],
                       help='Optimizer to use')
    parser.add_argument('--compare_all', action='store_true',
                       help='Train with all optimizers for comparison')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset = load_data(
        config.data_dir,
        config.block_size,
        config.train_split
    )
    
    # Update vocab size in config
    config.vocab_size = train_dataset.vocab_size
    
    # Determine which optimizers to train
    if args.compare_all:
        optimizers_to_train = ['sgd', 'momentum', 'adagrad', 'adam']
    else:
        optimizers_to_train = [args.optimizer]
    
    # Train with each optimizer
    for opt_name in optimizers_to_train:
        print(f"\n{'#'*60}")
        print(f"# Training with {opt_name.upper()}")
        print(f"{'#'*60}\n")
        
        # Create fresh model for each optimizer
        model_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
        )

        model = GPT(model_config)
        
        # Create optimizer
        optimizer = get_optimizer(model, opt_name, config)
        
        # Override learning rate if specified
        if args.lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        # Train
        experiment_name = opt_name
        train(model, train_dataset, val_dataset, optimizer, config, experiment_name)
    
    # Compare all optimizers if training multiple
    if args.compare_all:
        from utils import compare_optimizers, print_summary_statistics
        
        print("\n" + "="*60)
        print("Generating comparison plots...")
        comparison_plot_path = os.path.join(config.log_dir, "optimizer_comparison.png")
        compare_optimizers(config.log_dir, optimizers_to_train, comparison_plot_path)
        
        print_summary_statistics(config.log_dir, optimizers_to_train)


if __name__ == '__main__':
    main()
