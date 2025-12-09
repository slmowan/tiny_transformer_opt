"""
Training utilities including learning rate scheduling, gradient clipping, and evaluation
"""

import torch
import torch.nn as nn
import math


class LRScheduler:
    """
    Learning rate scheduler with warmup and cosine decay
    """
    
    def __init__(self, optimizer, warmup_steps, decay_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _get_lr(self):
        """Calculate current learning rate"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        elif self.current_step < self.decay_steps:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Minimum learning rate
            lr = self.min_lr
        
        return lr
    
    def get_last_lr(self):
        """Return current learning rate"""
        return [self._get_lr()]


def clip_gradients(model, max_norm):
    """
    Clip gradients by global norm
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
        
    Returns:
        total_norm: Total gradient norm before clipping
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Calculate total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in parameters])
    ).item()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    return total_norm


def compute_gradient_norm(model):
    """
    Compute the global gradient norm
    
    Args:
        model: Model to compute gradient norm for
        
    Returns:
        norm: Gradient norm
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in parameters])
    ).item()
    
    return total_norm


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, num_batches=50):
    """
    Evaluate model on dataset
    
    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate
        
    Returns:
        avg_loss: Average loss
        perplexity: Perplexity
    """
    from .data_utils import get_batch
    
    model.eval()
    total_loss = 0.0
    
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    model.train()
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, step, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into
        filepath: Path to checkpoint
        device: Device to load checkpoint on
        
    Returns:
        step: Training step of checkpoint
        loss: Loss at checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from step {step} with loss {loss:.4f}")
    
    return step, loss


@torch.no_grad()
def generate_sample(model, dataset, prompt="The ", max_tokens=200, temperature=0.8, device='cpu'):
    """
    Generate text sample from model
    
    Args:
        model: Model to generate from
        dataset: Dataset (for encoding/decoding)
        prompt: Starting text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run generation on
        
    Returns:
        generated_text: Generated text string
    """
    model.eval()
    
    # Encode prompt
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate
    generated_idx = model.generate(idx, max_tokens, temperature=temperature)
    
    # Decode
    generated_text = dataset.decode(generated_idx[0].tolist())
    
    model.train()
    return generated_text
