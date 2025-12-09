"""
Data loading and preprocessing utilities for Tiny Shakespeare dataset
"""

import torch
import requests
import os


class CharDataset:
    """Character-level dataset for language modeling"""
    
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        """Get a sequence and its target (next character prediction)"""
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def encode(self, text):
        """Encode text to indices"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices):
        """Decode indices to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices])


def download_tiny_shakespeare(data_dir='./data'):
    """Download Tiny Shakespeare dataset"""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'shakespeare.txt')
    
    if os.path.exists(filepath):
        print("Tiny Shakespeare dataset already downloaded.")
        return filepath
    
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print("Downloading Tiny Shakespeare dataset...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Dataset downloaded successfully to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def load_data(data_dir='./data', seq_len=256, train_split=0.9):
    """
    Load and prepare Tiny Shakespeare dataset
    
    Args:
        data_dir: Directory to store/load data
        seq_len: Sequence length for training
        train_split: Fraction of data for training
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    filepath = download_tiny_shakespeare(data_dir)
    
    # Read text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Total characters in dataset: {len(text)}")
    
    # Split into train and validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    train_dataset = CharDataset(train_text, seq_len)
    val_dataset = CharDataset(val_text, seq_len)
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_batch(dataset, batch_size, device='cpu'):
    """
    Generate a random batch from dataset
    
    Args:
        dataset: Dataset to sample from
        batch_size: Number of sequences in batch
        device: Device to place tensors on
        
    Returns:
        x: Input sequences (batch_size, seq_len)
        y: Target sequences (batch_size, seq_len)
    """
    indices = torch.randint(len(dataset), (batch_size,))
    x_list, y_list = [], []
    
    for idx in indices:
        x, y = dataset[idx.item()]
        x_list.append(x)
        y_list.append(y)
    
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    
    return x, y
