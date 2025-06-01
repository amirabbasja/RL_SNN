import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import math
import utils
from natsort import natsorted

def parseModelParams(filename_stem):
    """
    Parse model parameters from filename based on the pattern:
    
    Args:
        filename_stem (str): The filename without extension
    
    Returns:
        dict: Dictionary containing parsed parameters
    """
    
    parts = filename_stem.split('_')
    modelName = parts[0]
    runNumber = parts[-1]
    extraInfo = parts[-2]
    numENVS = parts[-3]
    gamma = parts[-4]
    miniBatchSize = parts[-5]
    eDecay = parts[-6]
    learningrate = parts[-7]
    hiddenNodes = parts[1:-7]

    return {
        "modelName":modelName,
        "runNumber": runNumber,
        "extraInfo": extraInfo,
        "numENVS": numENVS,
        "gamma": gamma,
        "miniBatchSize": miniBatchSize,
        "eDecay": eDecay,
        "learningrate": learningrate,
        "hiddenNodes": hiddenNodes,
    }


def plot_training_histories(directory_path):
    """
    Reads all .pth files from a directory and plots their training histories.
    
    Args:
        directory_path (str): Path to the directory containing .pth files
    """
    # Convert to Path object for easier handling
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Find all .pth files in the directory
    pth_files = natsorted(list(dir_path.glob("*.pth")))
    
    if not pth_files:
        print(f"No .pth files found in {directory_path}")
        return
    
    print(f"Found {len(pth_files)} .pth files")
    
    # Calculate subplot layout (max 2 columns)
    n_files = len(pth_files)
    n_cols = min(2, n_files)
    n_rows = math.ceil(n_files / n_cols)
    
    # Create figure and subplots with more space for parameter info
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7 * n_rows))
    
    # Handle case where there's only one subplot
    if n_files == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Process each file
    for idx, pth_file in enumerate(pth_files):
        # Parse model parameters from filename
        model_info = parseModelParams(pth_file.stem)
        
        try:
            # Load the .pth file
            print(f"Processing {pth_file.name}...")
            checkpoint = torch.load(pth_file, map_location='cpu', weights_only = False)
            
            # Extract training history
            if 'train_history' not in checkpoint:
                print(f"Warning: 'train_history' not found in {pth_file.name}")
                continue
                
            lst_history = checkpoint['train_history']
            
            if not lst_history:
                print(f"Warning: Empty training history in {pth_file.name}")
                continue
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(lst_history)
            df["points"] = df["points"].apply(lambda x: x[0]) 

            # Plot on the corresponding subplot
            ax = axes[idx]
            
            # Plot points vs episodes
            ax.plot(df['episode'], df['points'], linewidth=1)
            ax.set_title(f"Run {model_info['runNumber']} - {checkpoint['elapsedTime']}s", fontsize=12, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Points')
            ax.grid(True, alpha=0.3)
            
            # Add some statistics to the plot
            max_points = df['points'].max()
            min_points = df['points'].min()
            avg_points = df['points'].mean()
            
            # Add text box with statistics and parameters
            stats_text = f"Max: {max_points:.2f}\nMin: {min_points:.2f}\nAvg: {avg_points:.2f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            runDetails = f"Envs: {model_info['numENVS']} | Gamma: {model_info['gamma']} | Batch: {model_info['miniBatchSize']} | Decay {model_info['eDecay']} | lr: {model_info['learningrate']} | Hidden {model_info['hiddenNodes']}"
            # Add parameter information on the right side
            ax.text(0.98, 0.98, runDetails, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   fontsize=9)
            
        except Exception as e:
            print(f"Error processing {pth_file.name}: {str(e)}")
            # Create empty subplot with error message
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error loading\n{pth_file.name}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{pth_file.stem} (Error)", fontsize=12)
    
    # Hide any unused subplots
    for idx in range(len(pth_files), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.suptitle(f"Training Histories from {dir_path.name}", fontsize=16, y=0)
    plt.show()
    os.makedirs("./Data", exist_ok = True)
    plt.savefig("./Data/savedFig.png")
    
    return fig

def main():
    """
    Main function to run the script with user input or default directory.
    """
    # Prompt user for directory path
    directory_path = "./Data/parallelDQN_64_64_0.0001_0.9995_1000_0.995_2_"
    
    # Remove quotes if present
    if directory_path.startswith('"') and directory_path.endswith('"'):
        directory_path = directory_path[1:-1]
    
    plot_training_histories(directory_path)

if __name__ == "__main__":
    main()