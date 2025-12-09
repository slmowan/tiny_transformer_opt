"""
Metrics logging and visualization utilities
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class MetricsLogger:
    """Logger for tracking training metrics"""
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': [],
            'gradient_norm': [],
            'steps': []
        }
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Save metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat()
        }
    
    def log(self, step, **kwargs):
        """
        Log metrics for current step
        
        Args:
            step: Training step
            **kwargs: Metric name-value pairs
        """
        self.metrics['steps'].append(step)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save(self):
        """Save metrics to JSON file"""
        data = {
            'metadata': self.metadata,
            'metrics': self.metrics
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load metrics from JSON file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.metadata = data['metadata']
                self.metrics = data['metrics']
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Metrics - {self.experiment_name}", fontsize=16)
        
        steps = self.metrics['steps']
        
        # Training loss
        if 'train_loss' in self.metrics and len(self.metrics['train_loss']) > 0:
            axes[0, 0].plot(steps[:len(self.metrics['train_loss'])], 
                           self.metrics['train_loss'], label='Train Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation metrics
        if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > 0:
            val_steps = [steps[i] for i in range(len(self.metrics['val_loss']))]
            axes[0, 1].plot(val_steps, self.metrics['val_loss'], 
                           label='Val Loss', color='orange')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity
        if 'val_perplexity' in self.metrics and len(self.metrics['val_perplexity']) > 0:
            val_steps = [steps[i] for i in range(len(self.metrics['val_perplexity']))]
            axes[1, 0].plot(val_steps, self.metrics['val_perplexity'], 
                           label='Val Perplexity', color='green')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].set_title('Validation Perplexity')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norm
        if 'gradient_norm' in self.metrics and len(self.metrics['gradient_norm']) > 0:
            axes[1, 1].plot(steps[:len(self.metrics['gradient_norm'])], 
                           self.metrics['gradient_norm'], 
                           label='Gradient Norm', color='red', alpha=0.6)
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def compare_optimizers(log_dir, optimizer_names, save_path=None):
    """
    Compare multiple optimizers on the same plot
    
    Args:
        log_dir: Directory containing log files
        optimizer_names: List of optimizer names to compare
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Optimizer Comparison", fontsize=16)
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for idx, optimizer_name in enumerate(optimizer_names):
        log_file = os.path.join(log_dir, f"{optimizer_name}.json")
        
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found for {optimizer_name}")
            continue
        
        with open(log_file, 'r') as f:
            data = json.load(f)
            metrics = data['metrics']
            steps = metrics['steps']
            color = colors[idx % len(colors)]
            
            # Training loss
            if 'train_loss' in metrics:
                axes[0, 0].plot(steps[:len(metrics['train_loss'])], 
                               metrics['train_loss'], 
                               label=optimizer_name, color=color, alpha=0.7)
            
            # Validation loss
            if 'val_loss' in metrics:
                val_steps = [steps[i] for i in range(len(metrics['val_loss']))]
                axes[0, 1].plot(val_steps, metrics['val_loss'], 
                               label=optimizer_name, color=color, alpha=0.7)
            
            # Perplexity
            if 'val_perplexity' in metrics:
                val_steps = [steps[i] for i in range(len(metrics['val_perplexity']))]
                axes[1, 0].plot(val_steps, metrics['val_perplexity'], 
                               label=optimizer_name, color=color, alpha=0.7)
            
            # Gradient norm
            if 'gradient_norm' in metrics:
                axes[1, 1].plot(steps[:len(metrics['gradient_norm'])], 
                               metrics['gradient_norm'], 
                               label=optimizer_name, color=color, alpha=0.5)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title('Validation Perplexity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Norm')
    axes[1, 1].set_title('Gradient Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_statistics(log_dir, optimizer_names):
    """
    Print summary statistics for all optimizers
    
    Args:
        log_dir: Directory containing log files
        optimizer_names: List of optimizer names
    """
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*80)
    
    for optimizer_name in optimizer_names:
        log_file = os.path.join(log_dir, f"{optimizer_name}.json")
        
        if not os.path.exists(log_file):
            continue
        
        with open(log_file, 'r') as f:
            data = json.load(f)
            metrics = data['metrics']
            
            print(f"\n{optimizer_name.upper()}")
            print("-" * 40)
            
            if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
                final_train_loss = metrics['train_loss'][-1]
                print(f"  Final Training Loss:     {final_train_loss:.4f}")
            
            if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
                final_val_loss = metrics['val_loss'][-1]
                best_val_loss = min(metrics['val_loss'])
                print(f"  Final Validation Loss:   {final_val_loss:.4f}")
                print(f"  Best Validation Loss:    {best_val_loss:.4f}")
            
            if 'val_perplexity' in metrics and len(metrics['val_perplexity']) > 0:
                final_perplexity = metrics['val_perplexity'][-1]
                best_perplexity = min(metrics['val_perplexity'])
                print(f"  Final Perplexity:        {final_perplexity:.2f}")
                print(f"  Best Perplexity:         {best_perplexity:.2f}")
            
            if 'gradient_norm' in metrics and len(metrics['gradient_norm']) > 0:
                avg_grad_norm = np.mean(metrics['gradient_norm'])
                std_grad_norm = np.std(metrics['gradient_norm'])
                print(f"  Avg Gradient Norm:       {avg_grad_norm:.4f} ± {std_grad_norm:.4f}")
    
    print("\n" + "="*80)
