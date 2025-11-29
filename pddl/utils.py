"""
Utility functions for visualization and analysis
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_loss(losses_dict, path, info=["B.C", "I.C", "INT", "P.D.E"]):
    """Plot individual loss components"""
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(4), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

def plot_comprehensive_loss(losses_dict, path):
    """Plot comprehensive loss analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (key, losses) in enumerate(losses_dict.items()):
        axes[i].semilogy(losses)
        axes[i].set_title(f'{key.upper()} Loss')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_weight_history(weight_history, path):
    """Plot adaptive weight evolution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, weights in weight_history.items():
        ax.plot(weights, label=key.upper())
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title('Adaptive Loss Weights Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()