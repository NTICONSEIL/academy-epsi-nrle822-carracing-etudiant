"""
utils/viz.py
------------
Visualisations communes aux notebooks du projet fil rouge.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional
import torch


def show_frames_grid(frames: np.ndarray, n_cols: int = 4, title: str = "Frames CarRacing-v2") -> None:
    """Affiche une grille de frames sous forme d'images."""
    n = min(len(frames), n_cols * 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(frames[i] if frames[i].ndim == 3 else frames[i, :, :, 0], cmap="gray" if frames[i].ndim == 2 else None)
            ax.set_title(f"Frame {i+1}", fontsize=9)
        ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def show_feature_maps(feature_maps: torch.Tensor, original_frame: Optional[np.ndarray] = None,
                      title: str = "Feature maps") -> None:
    """
    Affiche les feature maps d'une couche convolutionnelle.
    
    feature_maps : tenseur (C, H, W) — cartes d'activation
    original_frame : image originale (optionnel, affichée en premier)
    """
    fmaps = feature_maps.cpu().detach().numpy()
    n_filters = fmaps.shape[0]
    n_cols = min(8, n_filters + (1 if original_frame is not None else 0))
    n_rows = (n_filters + (1 if original_frame is not None else 0) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    axes = np.array(axes).flatten()

    offset = 0
    if original_frame is not None:
        axes[0].imshow(original_frame)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")
        offset = 1

    for i in range(n_filters):
        ax = axes[i + offset]
        ax.imshow(fmaps[i], cmap="RdBu_r")
        ax.set_title(f"Filtre {i+1}", fontsize=9)
        ax.axis("off")

    for ax in axes[n_filters + offset:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_episode_summary(rewards: List[float], steerings: List[float],
                          title: str = "Résumé d'épisode") -> None:
    """Affiche un résumé d'épisode : récompenses cumulées et direction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = range(len(rewards))

    cumulative = np.cumsum(rewards)
    axes[0].plot(steps, cumulative, color="steelblue", lw=1.5)
    axes[0].fill_between(steps, 0, cumulative, alpha=0.15, color="steelblue")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Récompense cumulée")
    axes[0].set_title("Récompense cumulée"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, steerings, color="coral", lw=1, alpha=0.8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].fill_between(steps, 0, steerings,
                         where=np.array(steerings) > 0, alpha=0.2, color="coral", label="Droite")
    axes[1].fill_between(steps, 0, steerings,
                         where=np.array(steerings) < 0, alpha=0.2, color="steelblue", label="Gauche")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Steering")
    axes[1].set_title("Direction au fil du temps"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
