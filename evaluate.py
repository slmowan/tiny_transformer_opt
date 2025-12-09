"""
Evaluation script for analyzing trained models
Generates text samples and computes metrics
"""

import torch
import argparse
import os

from config import Config
from model import TinyTransformerLM
from optimizers import SGD, MomentumSGD, Adagrad, Adam
from utils import load_data, evaluate, generate_sample


def load_trained_model(checkpoint_path, config, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        device: Device to load model on
        
    Returns:
        model: Loaded model
        step: Training step of checkpoint
    """
    model = TinyTransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    step = checkpoint['step']
    
    return model, step


def generate_multiple_samples(model, dataset, prompts, max_tokens=200, temperature=0.8, device='cpu'):
    """
    Generate text from multiple prompts
    
    Args:
        model: Model to generate from
        dataset: Dataset for encoding/decoding
        prompts: List of starting prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on
    """
    print("\n" + "="*60)
    print("GENERATED SAMPLES")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: \"{prompt}\"")
        print("-" * 60)
        
        sample = generate_sample(
            model, dataset, prompt, max_tokens, temperature, device
        )
        
        print(sample)
        print()


def evaluate_all_checkpoints(config, dataset, device='cpu'):
    """
    Evaluate all saved checkpoints
    
    Args:
        config: Configuration object
        dataset: Validation dataset
        device: Device to run evaluation on
    """
    checkpoint_dir = config.checkpoint_dir
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_final.pt')]
    
    if not checkpoint_files:
        print("No final checkpoints found!")
        return
    
    print("\n" + "="*60)
    print("CHECKPOINT EVALUATION")
    print("="*60)
    
    results = []
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        optimizer_name = checkpoint_file.replace('_final.pt', '')
        
        print(f"\nEvaluating {optimizer_name}...")
        
        model, step = load_trained_model(checkpoint_path, config, device)
        val_loss, val_perplexity = evaluate(
            model, dataset, config.batch_size, device, num_batches=100
        )
        
        results.append({
            'optimizer': optimizer_name,
            'step': step,
            'val_loss': val_loss,
            'perplexity': val_perplexity
        })
        
        print(f"  Step: {step}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Perplexity: {val_perplexity:.2f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Sort by perplexity
    results.sort(key=lambda x: x['perplexity'])
    
    print(f"\n{'Rank':<6} {'Optimizer':<15} {'Val Loss':<12} {'Perplexity':<12}")
    print("-" * 60)
    
    for rank, result in enumerate(results, 1):
        print(f"{rank:<6} {result['optimizer']:<15} {result['val_loss']:<12.4f} {result['perplexity']:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint to evaluate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'momentum', 'adagrad', 'adam'],
                       help='Optimizer name (used if checkpoint not specified)')
    parser.add_argument('--evaluate_all', action='store_true',
                       help='Evaluate all final checkpoints')
    parser.add_argument('--generate', action='store_true',
                       help='Generate text samples')
    parser.add_argument('--prompts', type=str, nargs='+',
                       default=['ROMEO:', 'The king ', 'What is '],
                       help='Prompts for text generation')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    device = config.device
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset = load_data(
        config.data_dir,
        config.max_seq_len,
        config.train_split
    )
    config.vocab_size = train_dataset.vocab_size
    
    # Evaluate all checkpoints
    if args.evaluate_all:
        evaluate_all_checkpoints(config, val_dataset, device)
        return
    
    # Load specific checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            f"{args.optimizer}_final.pt"
        )
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model, step = load_trained_model(checkpoint_path, config, device)
    
    print(f"Model loaded from step {step}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_perplexity = evaluate(
        model, val_dataset, config.batch_size, device, num_batches=100
    )
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {val_perplexity:.2f}")
    
    # Generate samples
    if args.generate:
        generate_multiple_samples(
            model, train_dataset, args.prompts,
            args.max_tokens, args.temperature, device
        )


if __name__ == '__main__':
    main()
