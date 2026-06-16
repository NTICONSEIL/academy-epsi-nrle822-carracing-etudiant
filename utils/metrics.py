"""
utils/metrics.py
----------------
Fonctions d'évaluation communes aux 4 notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Métriques de régression et classification
# ──────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur quadratique moyenne (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur absolue moyenne (Mean Absolute Error)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Précision directionnelle pour le steering.
    
    Mesure si le modèle prédit correctement la direction
    (gauche / tout droit / droite), indépendamment de l'amplitude.
    
    Seuil : steering > 0.1 → droite, < -0.1 → gauche, sinon → tout droit
    """
    def to_direction(x):
        return np.where(x > 0.1, 1, np.where(x < -0.1, -1, 0))
    
    dir_true = to_direction(y_true)
    dir_pred = to_direction(y_pred)
    return float(np.mean(dir_true == dir_pred))


def print_regression_report(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Steering") -> None:
    """Affiche un rapport de métriques de régression."""
    print(f"{'='*40}")
    print(f"📊 Rapport d'évaluation — {label}")
    print(f"{'='*40}")
    print(f"  RMSE               : {rmse(y_true, y_pred):.4f}")
    print(f"  MAE                : {mae(y_true, y_pred):.4f}")
    print(f"  Précision directionnelle : {directional_accuracy(y_true, y_pred)*100:.1f}%")
    print(f"  Biais moyen (mean error) : {float(np.mean(y_pred - y_true)):.4f}")
    print(f"{'='*40}")


# ──────────────────────────────────────────────
# Courbes d'entraînement
# ──────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Courbes d'apprentissage",
    ylabel: str = "Loss (MSE)",
    figsize: tuple = (10, 4),
) -> None:
    """
    Affiche les courbes de loss train et validation.
    
    Interprétation visuelle :
    - Train et val proches → bon ajustement
    - Train << val → surapprentissage (overfitting)
    - Train et val élevés → sous-apprentissage (underfitting)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Courbe complète
    axes[0].plot(epochs, train_losses, "b-", label="Train", linewidth=1.5)
    axes[0].plot(epochs, val_losses, "r--", label="Validation", linewidth=1.5)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} (complète)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Courbe après warm-up (ignorer les premières époques)
    warmup = max(1, len(train_losses) // 10)
    axes[1].plot(epochs[warmup:], train_losses[warmup:], "b-", label="Train", linewidth=1.5)
    axes[1].plot(epochs[warmup:], val_losses[warmup:], "r--", label="Validation", linewidth=1.5)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"{title} (après warm-up)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Diagnostic automatique
    final_gap = val_losses[-1] - train_losses[-1]
    if final_gap > 0.1 * val_losses[-1]:
        print("⚠️  Diagnostic : surapprentissage détecté (val >> train). "
              "Essayez Dropout, régularisation L2, ou plus de données.")
    elif train_losses[-1] > 0.5:
        print("⚠️  Diagnostic : sous-apprentissage possible. "
              "Essayez un réseau plus profond ou un learning rate plus élevé.")
    else:
        print("✅ Diagnostic : convergence correcte.")


def plot_reward_curve(
    episode_rewards: List[float],
    window: int = 10,
    title: str = "Récompenses par épisode (DQN)",
) -> None:
    """
    Affiche la courbe de récompense de l'agent DQN avec moyenne mobile.
    
    La moyenne mobile (rolling mean) lisse les variations épisodiques
    et révèle la tendance d'apprentissage.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    episodes = range(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, alpha=0.3, color="steelblue", label="Récompense brute")
    
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode="valid"
        )
        ax.plot(
            range(window, len(episode_rewards) + 1),
            moving_avg,
            color="steelblue",
            linewidth=2,
            label=f"Moyenne mobile ({window} épisodes)"
        )
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    if len(episode_rewards) >= window:
        print(f"📊 Récompense moyenne (derniers {window} épisodes) : "
              f"{np.mean(episode_rewards[-window:]):.1f}")


def plot_predictions_vs_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 200,
    title: str = "Prédictions vs Réalité — Steering",
) -> None:
    """
    Compare les prédictions du modèle avec les actions réelles.
    Affiche les 2 premières colonnes si y est multidimensionnel.
    """
    y_true = np.array(y_true).flatten()[:n_samples]
    y_pred = np.array(y_pred).flatten()[:n_samples]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Série temporelle
    t = range(len(y_true))
    axes[0].plot(t, y_true, label="Réalité", alpha=0.7, linewidth=1)
    axes[0].plot(t, y_pred, label="Prédiction", alpha=0.7, linewidth=1, linestyle="--")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Valeur")
    axes[0].set_title(f"{title} — Série temporelle")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", label="Prédiction parfaite")
    axes[1].set_xlabel("Valeur réelle")
    axes[1].set_ylabel("Valeur prédite")
    axes[1].set_title(f"{title} — Scatter")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Comparaison de modèles
# ──────────────────────────────────────────────

def compare_models(results: Dict[str, Dict]) -> None:
    """
    Compare plusieurs modèles sur les métriques clés.
    
    Paramètres
    ----------
    results : dict de la forme
        {
            "CNN baseline" : {"rmse": 0.12, "dir_acc": 0.78, "val_loss": 0.021},
            "CNN + LSTM"   : {"rmse": 0.09, "dir_acc": 0.84, "val_loss": 0.015},
        }
    
    Exemple
    -------
    >>> compare_models({
    ...     "MLP seul": {"rmse": 0.20, "dir_acc": 0.65, "val_loss": 0.08},
    ...     "CNN baseline": {"rmse": 0.12, "dir_acc": 0.78, "val_loss": 0.021},
    ... })
    """
    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(model_names)))
    
    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(metric.upper().replace("_", " "))
        ax.set_ylabel("Valeur")
        
        # Annoter les barres
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9
            )
        
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Comparaison des modèles", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
