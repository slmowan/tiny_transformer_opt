"""
Configuration file for Tiny LLM training
Contains all hyperparameters and settings for the experiments
"""

import torch

class Config:
    # Model architecture
    vocab_size = None  # Will be set after loading dataset
    n_layers = 2
    n_heads = 4
    d_model = 128
    d_ff = 512  # Feed-forward dimension (typically 4x d_model)
    dropout = 0.1
    max_seq_len = 256
    
    # Training settings
    batch_size = 64
    num_epochs = 50
    eval_interval = 100  # Evaluate every N steps
    save_interval = 1000  # Save checkpoint every N steps
    
    # Optimizer settings
    learning_rate = 0.001
    weight_decay = 0.01
    betas = (0.9, 0.999)  # For Adam
    momentum = 0.9  # For Momentum SGD
    eps = 1e-8  # For Adam and Adagrad
    
    # Learning rate scheduling
    use_lr_schedule = True
    warmup_steps = 500
    lr_decay_steps = 10000
    min_lr = 1e-5
    
    # Gradient clipping
    use_grad_clip = True
    max_grad_norm = 1.0
    
    # Data settings
    train_split = 0.9
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    
    # Logging
    log_interval = 10  # Log every N steps
    
    # Seed for reproducibility
    seed = 42
    
    @staticmethod
    def get_optimizer_configs():
        """
        Returns a dictionary of optimizer-specific configurations
        """
        return {
            'sgd': {
                'lr': 0.01,
                'weight_decay': 0.0
            },
            'momentum': {
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0
            },
            'adagrad': {
                'lr': 0.01,
                'eps': 1e-8,
                'weight_decay': 0.0
            },
            'adam': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.01
            }
        }
