import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(metrics, title="Training Rewards", filename="rewards.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['rewards'], alpha=0.5)
    
    window = min(50, len(metrics['rewards']))
    if window > 0:
        smoothed = [sum(metrics['rewards'][i:i+window])/window for i in range(len(metrics['rewards'])-window)]
        plt.plot(range(window, len(metrics['rewards'])), smoothed, color='red', label='Moving Average')
        
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def plot_success_rate(metrics, title="Success Rate", filename="success.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['success_rate'])
    plt.title(title)
    plt.xlabel("Episode/Epoch")
    plt.ylabel("Success Rate")
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def plot_heatmap(visitation_matrix, title="Agent Path Heatmap", filename="heatmap.png"):
    """Phase 7 (Step 29): Path Heatmap visualization"""
    plt.figure(figsize=(8, 6))
    # Standard Matplotlib heatmap (without needing seaborn dependency)
    plt.imshow(visitation_matrix, cmap="hot", interpolation='nearest')
    plt.colorbar(label="Visit Count")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def plot_comparison(metrics_dict, metric_key='rewards', title="Agent Comparison", filename="comparison.png"):
    """Phase 7 (Step 30): Compare multiple agents"""
    plt.figure(figsize=(10, 5))
    
    for agent_name, metrics in metrics_dict.items():
        if metric_key in metrics:
            data = metrics[metric_key]
            window = min(20, len(data))
            if window > 0:
                smoothed = [sum(data[i:i+window])/window for i in range(len(data)-window)]
                plt.plot(range(window, len(data)), smoothed, label=f"{agent_name} (Smoothed)")
            else:
                plt.plot(data, label=agent_name)
    
    plt.title(title)
    plt.xlabel("Episode / Epoch")
    plt.ylabel(metric_key.capitalize())
    plt.legend()
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
