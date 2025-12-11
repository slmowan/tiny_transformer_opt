import json
import matplotlib.pyplot as plt
import os

# ================= Configuration =================
# Directory containing the json files
FOLDER_NAME = 'plot'

# Mapping identifiers to filenames
# We will try to read the real optimizer name from the JSON, 
# but we use this map to find the files.
files_map = {
    'Adagrad': 'lr_metrics_summary_adagrad.json',
    'Adam': 'lr_metrics_summary_adam.json',
    'AdamW': 'lr_metrics_summary_adamw.json',
    'Momentum': 'lr_metrics_summary_momentum.json',
    'SGD': 'lr_metrics_summary_sgd.json'
}

# Key names configuration based on your description
KEYS = {
    'train': 'train_losses',
    'val': 'val_losses',
    'opt_name': 'optimizer',
    'lr': 'learning_rate'
}
# ===============================================

def get_data_from_json(filepath, default_name):
    """
    Reads the JSON file and extracts loss lists.
    Returns: (label_name, train_steps, train_losses, val_steps, val_losses)
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return default_name, [], [], [], []

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}")
        return default_name, [], [], [], []

    # 1. Extract Metadata for Label (Optimizer Name + LR)
    opt_name = data.get(KEYS['opt_name'], default_name)
    lr = data.get(KEYS['lr'], None)
    
    # Construct a label like "Adam (lr=1e-3)" or just "Adam"
    if lr is not None:
        label = f"{opt_name} (lr={lr})"
    else:
        label = f"{opt_name}"

    # 2. Extract Training Data
    t_losses = data.get(KEYS['train'], [])
    # Generate linear steps for X-axis [1, 2, 3, ... len]
    t_steps = list(range(1, len(t_losses) + 1))

    # 3. Extract Validation Data
    v_losses = data.get(KEYS['val'], [])
    v_steps = list(range(1, len(v_losses) + 1))

    return label, t_steps, t_losses, v_steps, v_losses

def plot_comparison(data_list, title, ylabel, filename_suffix):
    """
    Plots the comparison chart.
    data_list: list of tuples (label, steps, losses)
    """
    plt.figure(figsize=(12, 7))
    
    has_data = False
    for label, steps, losses in data_list:
        if len(losses) > 0:
            plt.plot(steps, losses, label=label, alpha=0.8, linewidth=1.5)
            has_data = True
        else:
            print(f"Info: No {ylabel} data for '{label}'")

    if not has_data:
        print(f"Warning: No data found for {ylabel} plot. Skipping.")
        plt.close()
        return

    plt.title(title, fontsize=16)
    plt.xlabel('Steps / Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure into the plot folder
    out_filename = f"optimizer_comparison_{filename_suffix}.png"
    out_path = os.path.join(FOLDER_NAME, out_filename)
    
    plt.savefig(out_path, dpi=300)
    print(f"Figure saved: {out_path}")
    
    # Close to free memory
    plt.close()

def main():
    # Lists to store plotting data
    # Structure: [(label, steps, losses), ...]
    train_plot_data = []
    val_plot_data = []

    print("Starting data processing...")
    
    # Iterate through the files defined in files_map
    for default_name, filename in files_map.items():
        path = os.path.join(FOLDER_NAME, filename)
        
        # Extract data
        label, t_steps, t_losses, v_steps, v_losses = get_data_from_json(path, default_name)
        
        # Store valid data
        train_plot_data.append((label, t_steps, t_losses))
        val_plot_data.append((label, v_steps, v_losses))

    # Plot 1: Training Loss
    print("Plotting Training Loss...")
    plot_comparison(
        train_plot_data, 
        title='Training Loss Comparison', 
        ylabel='Training Loss', 
        filename_suffix='train_loss'
    )

    # Plot 2: Validation Loss
    print("Plotting Validation Loss...")
    plot_comparison(
        val_plot_data, 
        title='Validation Loss Comparison', 
        ylabel='Validation Loss', 
        filename_suffix='val_loss'
    )

    print("Done.")

if __name__ == "__main__":
    main()