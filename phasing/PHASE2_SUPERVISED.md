# Phase 2 — Imitation d'un pilote (Apprentissage supervisé)
## NRLE822 · Projet fil rouge · CarRacing-v2

> **Phase** 2/4 &nbsp;|&nbsp; **Durée** 4h (Séance S3 — FFP) &nbsp;|&nbsp; **Notebook** `02_Supervised_imitation.ipynb`  
> **Compétences** CDPEIA 2.5 · 2.6 · 2.7 &nbsp;|&nbsp; **Prérequis** Phase 1 complétée

---

## Objectif

Entraîner un modèle à **prédire les actions de conduite** (steering, gas, brake) à partir des observations, en imitant un pilote rule-based dont les données sont fournies dans les fichiers CSV du repo.

À la fin de cette phase vous saurez :

- Construire une pipeline d'apprentissage supervisé complète pour des données séquentielles
- Comprendre et appliquer la rétropropagation avec une loss de régression
- Effectuer un split temporel correct (sans data leakage)
- Évaluer un modèle avec des métriques adaptées au problème

---

## 2.1 Rappels théoriques

### La rétropropagation (backpropagation)

L'algorithme d'entraînement d'un réseau de neurones repose sur 4 étapes répétées à chaque batch :

```
1. Forward pass  : ŷ = modèle(x)          → calculer la prédiction
2. Loss          : L = loss(ŷ, y)          → mesurer l'écart avec la vérité
3. Backward pass : ∂L/∂w pour chaque w    → calculer les gradients
4. Update        : w ← w − lr × ∂L/∂w    → mettre à jour les poids
```

Le gradient ∂L/∂w est calculé par la **règle de la chaîne** en remontant couche par couche depuis la sortie vers l'entrée — d'où le nom « rétropropagation ».

### Régression vs classification

En Phase 1, vous avez fait de la **classification** : prédire une étiquette parmi 3 classes (droite / gauche / tout droit). La loss utilisée était `CrossEntropyLoss`.

En Phase 2, vous faites de la **régression** : prédire des valeurs continues.

| | Classification | Régression |
|--|--|--|
| **Sortie** | Probabilité par classe | Valeur réelle |
| **Loss** | CrossEntropyLoss | MSELoss (ou MAELoss) |
| **Activation finale** | Softmax (incluse dans CELoss) | Aucune (ou Tanh si sortie bornée) |
| **Métrique** | Accuracy (%) | RMSE, MAE |
| **Exemple Phase 1** | « C'est un virage gauche » | — |
| **Exemple Phase 2** | — | « steering = −0.43, gas = 0.61 » |

### MSE Loss

La **Mean Squared Error** pénalise les grandes erreurs plus fortement que les petites :

```
L = (1/n) × Σ (ŷᵢ − yᵢ)²
```

Elle est dérivable partout → compatible avec la descente de gradient. Pour la régression de valeurs de conduite, c'est le choix standard.

### L'optimiseur Adam

Adam adapte le learning rate **individuellement pour chaque paramètre** en maintenant une moyenne mobile des gradients passés. Il converge généralement plus vite que SGD et est moins sensible au choix du learning rate initial.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# lr=1e-3 est un bon point de départ
# Trop élevé → instabilité   |   Trop faible → convergence lente
```

---

## 2.2 Le dataset supervisé

### Structure des fichiers CSV

Les données se trouvent dans `data/demo_episodes/`. Chaque fichier correspond à un épisode de conduite généré par un agent rule-based.

```
data/demo_episodes/
├── metadata.csv          ← résumé des épisodes (id, steps, reward)
├── episode_000.csv       ← ~430 transitions
├── episode_001.csv
└── episode_002.csv
```

Chaque ligne d'un CSV = un pas de temps (step) :

| Colonne | Type | Description |
|---------|------|-------------|
| `step` | int | Numéro du pas de temps |
| `obs_mean_r` | float | Moyenne du canal R de l'observation (0–255) |
| `obs_mean_g` | float | Moyenne du canal G |
| `obs_mean_b` | float | Moyenne du canal B |
| `road_offset` | float | Position latérale de la route (−1 = gauche, +1 = droite) |
| `action_steering` | float | Direction choisie (−1 = gauche, +1 = droite) |
| `action_gas` | float | Accélération (0–1) |
| `action_brake` | float | Freinage (0–1) |
| `reward` | float | Récompense reçue à ce step |
| `cumulative_reward` | float | Récompense cumulée depuis le début de l'épisode |
| `terminated` | bool | True si l'épisode s'est terminé naturellement |

**Features (X) utilisées :** `[obs_mean_r, obs_mean_g, obs_mean_b, road_offset]`  
**Cibles (y) à prédire :** `[action_steering, action_gas, action_brake]`

> 💡 On utilise ici des features résumées plutôt que les frames brutes. C'est intentionnel pour cette phase (focus sur la pipeline supervisée). Les frames brutes seront réintroduites en Phase 3 via le LSTM.

---

## 2.3 Le point critique : le split temporel

C'est **le point le plus souvent raté** dans les projets de ML sur données séquentielles. Il sera explicitement vérifié dans la grille d'évaluation.

### Pourquoi le split classique ne fonctionne pas ici

Pour des données indépendantes (images de chats/chiens), on peut mélanger avant de splitter. Pour des données de conduite, **les frames consécutives sont fortement corrélées** : la frame 42 prédit presque parfaitement la frame 43.

Si on mélange les lignes avant de splitter :

```python
# ❌ FAUX — data leakage
df_shuffled = df.sample(frac=1, random_state=42)
train = df_shuffled[:800]
val   = df_shuffled[800:]
# Résultat : des frames n et n+1 se retrouvent dans train ET val
# Le modèle "mémorise" les transitions → métriques artificiellement bonnes
```

### Le split correct : par épisode entier

```python
# ✅ CORRECT — split par épisode
import pandas as pd, glob

csv_files = sorted(glob.glob('data/demo_episodes/episode_*.csv'))
dfs = [pd.read_csv(f) for f in csv_files]

# Épisodes 0 et 1 → train (données les plus riches)
train_df = pd.concat(dfs[:2], ignore_index=True)

# Épisode 2, première moitié → val
df_ep2 = dfs[2]
mid    = len(df_ep2) // 2
val_df  = df_ep2.iloc[:mid]
test_df = df_ep2.iloc[mid:]

print(f'Train : {len(train_df)} steps  (épisodes 0-1)')
print(f'Val   : {len(val_df)} steps   (ep 2, 1ère moitié)')
print(f'Test  : {len(test_df)} steps  (ep 2, 2ème moitié)')
```

> 💬 **Question** : montrez empiriquement la différence entre les deux approches. Entraînez le même modèle avec un split aléatoire puis avec un split temporel. Comparez les métriques sur le test set. Que constatez-vous ?

---

## 2.4 Architecture à implémenter

### `DrivingMLP`

```python
class DrivingMLP(nn.Module):
    """
    MLP pour l'imitation learning.
    Entrée  : 4 features résumées [R_moy, G_moy, B_moy, road_offset]
    Sortie  : 3 actions [steering, gas, brake]
    """
    def __init__(self, input_dim=4, hidden=128, output_dim=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            # Pas d'activation finale : MSELoss attend des valeurs réelles
        )

    def forward(self, x):
        return self.net(x)
```

> 💬 **Question** : pourquoi n'y a-t-il pas d'activation sur la dernière couche ? Que se passerait-il si on ajoutait un `Tanh` (dont la sortie est bornée ∈ [−1, 1]) ?

---

## 2.5 Travail à réaliser (notebook 02)

### Cellule 1 — Configuration

```python
import numpy as np, pandas as pd, glob, os, sys
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.append('..')
from utils.metrics import (rmse, mae, directional_accuracy,
                           plot_training_curves, plot_predictions_vs_truth,
                           print_regression_report)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
```

---

### Cellule 2 — Chargement et exploration du dataset

```python
DATA_DIR  = '../data/demo_episodes'
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'episode_*.csv')))

# TODO : Charger tous les CSV dans une liste de DataFrames
# dfs = [pd.read_csv(f) for f in csv_files]

# TODO : Afficher pour chaque épisode : nombre de steps et reward total

# TODO : Concaténer tous les épisodes et afficher :
#   - Les colonnes disponibles
#   - Les statistiques descriptives des colonnes action_*
#   - Un histogramme des 3 actions (steering, gas, brake)
```

> 💬 **Question** : la distribution du steering est-elle équilibrée ? Qu'est-ce que cela implique pour l'entraînement ?

---

### Cellule 3 — Split temporel et Dataset PyTorch

```python
class DrivingDataset(Dataset):
    FEATURES = ['obs_mean_r', 'obs_mean_g', 'obs_mean_b', 'road_offset']
    TARGETS   = ['action_steering', 'action_gas', 'action_brake']

    def __init__(self, df: pd.DataFrame):
        X = df[self.FEATURES].values.astype(np.float32)
        y = df[self.TARGETS].values.astype(np.float32)

        # TODO : Normaliser les couleurs (colonnes 0-2) vers [0, 1]
        # road_offset est déjà dans [-1, 1], pas besoin de normaliser

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# TODO : Implémenter le split temporel par épisode (cf. section 2.3)
# TODO : Créer train_ds, val_ds, test_ds
# TODO : Créer les DataLoaders (batch_size=64, shuffle=True pour train seulement)

print(f'Shape X : {train_ds.X.shape}  — 4 features par step')
print(f'Shape y : {train_ds.y.shape}  — 3 actions à prédire')
```

---

### Cellule 4 — Modèle `DrivingMLP`

```python
# TODO : Implémenter DrivingMLP (cf. section 2.4)

# Vérification obligatoire
model = DrivingMLP().to(DEVICE)
dummy = torch.zeros(8, 4).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Entrée {dummy.shape} → Sortie {out.shape}')  # Attendu: torch.Size([8, 3])
assert out.shape == (8, 3), 'Dimensions incorrectes !'

total = sum(p.numel() for p in model.parameters())
print(f'Paramètres : {total:,}')
```

---

### Cellule 5 — Boucle d'entraînement

```python
def train_mlp(model, train_ld, val_ld, n_epochs=30, lr=1e-3, device=DEVICE):
    """
    Entraîne le modèle et retourne l'historique des losses.
    Sauvegarde le meilleur modèle selon val_loss.
    """
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # ReduceLROnPlateau : divise lr par 2 si val_loss ne s'améliore pas
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    history  = {'tl': [], 'vl': []}
    best_vl  = float('inf')

    for ep in range(1, n_epochs + 1):
        # ── Phase train ──────────────────────────────────────────────────
        model.train()
        tl = 0.
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            # TODO : forward → loss → zero_grad → backward → step
            tl += loss.item()
        tl /= len(train_ld)

        # ── Phase val ────────────────────────────────────────────────────
        model.eval()
        vl = 0.
        with torch.no_grad():
            for xb, yb in val_ld:
                # TODO : forward uniquement, accumuler vl
                pass
        vl /= len(val_ld)
        sched.step(vl)

        history['tl'].append(tl)
        history['vl'].append(vl)

        if vl < best_vl:
            best_vl = vl
            torch.save(model.state_dict(), '/tmp/mlp_best.pt')

        if ep % 10 == 0:
            print(f'Ep {ep:3d} | Train={tl:.5f} | Val={vl:.5f} '
                  f'| LR={opt.param_groups[0]["lr"]:.6f}')

    model.load_state_dict(torch.load('/tmp/mlp_best.pt'))
    return history


model = DrivingMLP().to(DEVICE)
hist  = train_mlp(model, train_ld, val_ld, n_epochs=30)
```

---

### Cellule 6 — Courbes d'apprentissage

```python
# La fonction plot_training_curves de utils/metrics.py affiche les courbes
# et imprime un diagnostic automatique (surapprentissage / sous-apprentissage)
plot_training_curves(hist['tl'], hist['vl'],
                     title='DrivingMLP — Imitation Learning',
                     ylabel='MSE Loss')
```

---

### Cellule 7 — Évaluation sur le test set

```python
model.eval()
all_y, all_pred = [], []

with torch.no_grad():
    for xb, yb in test_ld:
        pred = model(xb.to(DEVICE)).cpu()
        all_y.append(yb.numpy())
        all_pred.append(pred.numpy())

y_true = np.vstack(all_y)    # (N, 3)
y_pred = np.vstack(all_pred) # (N, 3)

# TODO : Afficher le rapport pour le steering (colonne 0)
# print_regression_report(y_true[:, 0], y_pred[:, 0], label='Steering')

# TODO : Visualiser prédictions vs réalité pour le steering
# plot_predictions_vs_truth(y_true[:, 0], y_pred[:, 0])
```

---

### Cellule 8 — Comparaison avec la baseline naïve

Une baseline naïve prédit toujours la **moyenne d'entraînement**. Si votre modèle ne fait pas mieux, il n'a rien appris.

```python
# Calculer la moyenne d'entraînement pour chaque action
y_train_mean = train_df[['action_steering', 'action_gas', 'action_brake']].values.mean(axis=0)
y_baseline   = np.tile(y_train_mean, (len(y_true), 1))

print('=== Comparaison Baseline naïve vs DrivingMLP ===')
for i, action in enumerate(['Steering', 'Gas', 'Brake']):
    base_rmse = rmse(y_true[:, i], y_baseline[:, i])
    mlp_rmse  = rmse(y_true[:, i], y_pred[:, i])
    gain      = (base_rmse - mlp_rmse) / base_rmse * 100
    print(f'{action:10s} | Baseline={base_rmse:.4f} | MLP={mlp_rmse:.4f} | Gain={gain:.1f}%')
```

---

### Cellule 9 — Impact du learning rate

```python
# TODO : Entraîner le même modèle avec 3 valeurs de lr : 1e-2, 1e-3, 1e-4
# Comparer les val_loss finales et tracer les 3 courbes sur le même graphe

results_lr = {}
for lr in [1e-2, 1e-3, 1e-4]:
    m = DrivingMLP().to(DEVICE)
    h = train_mlp(m, train_ld, val_ld, n_epochs=20, lr=lr)
    results_lr[f'lr={lr:.0e}'] = {'val_loss': min(h['vl'])}
    print(f'lr={lr:.0e} → val_loss finale = {min(h["vl"]):.5f}')
```

> 💬 **Question** : quel learning rate donne la meilleure convergence ? Observez-vous de l'instabilité pour `lr=1e-2` ? Expliquez pourquoi.

---

## 2.6 Métriques d'évaluation

Le RMSE seul ne suffit pas pour évaluer un pilote. Trois métriques complémentaires sont disponibles dans `utils/metrics.py` :

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **RMSE** | `√( (1/n) Σ (ŷ−y)² )` | Amplitude moyenne des erreurs (mêmes unités que y) |
| **MAE** | `(1/n) Σ |ŷ−y|` | Plus robuste aux valeurs aberrantes que RMSE |
| **Précision directionnelle** | `mean(sign(ŷ) == sign(y))` | Est-ce que le modèle tourne dans le bon sens ? |

```python
from utils.metrics import rmse, mae, directional_accuracy, print_regression_report

# Rapport complet pour le steering
print_regression_report(y_true[:, 0], y_pred[:, 0], label='Steering (direction)')

# Précision directionnelle : prédire gauche/tout droit/droite correctement
dir_acc = directional_accuracy(y_true[:, 0], y_pred[:, 0])
print(f'Précision directionnelle : {dir_acc*100:.1f}%')
```

> 💡 Un modèle avec un bon RMSE mais une faible précision directionnelle tourne dans la bonne amplitude mais parfois dans le mauvais sens — catastrophique pour la conduite !

---

## 2.7 Questions de réflexion

À rédiger dans des **cellules Markdown** du notebook :

1. Votre modèle améliore-t-il significativement la baseline naïve sur les 3 actions ? Sur laquelle est-il le moins performant ? Pourquoi ?

2. La distribution du steering dans le dataset est asymétrique (beaucoup de « tout droit »). Comment cela affecte-t-il les métriques ? Proposez une solution (pondération des samples, sur-échantillonnage des virages…).

3. Comparez le RMSE et la précision directionnelle de votre meilleur modèle. Sont-ils cohérents ? Lequel est le plus pertinent pour évaluer un pilote autonome ?

4. *(Optionnel)* Testez une architecture différente : ajoutez une couche, changez la taille de la couche cachée, ou retirez le Dropout. Comparez les résultats avec votre modèle de base.

---

## 2.8 Limitation et transition vers la Phase 3

> ⚠️ **Ce que la Phase 2 ne peut pas faire**
>
> Votre MLP prédit chaque action de manière **indépendante** de ce qui précède : il n'a aucune mémoire. Pour prédire l'action au step 100, il ne « sait » pas que les 10 steps précédents étaient un virage en courbe — il ne voit que la frame courante.
>
> **Conséquence concrète** : pour un virage qui s'accentue progressivement, le modèle réagit trop tard, comme un conducteur qui regarderait la route avec des œillères.
>
> La Phase 3 va corriger ce problème : en donnant au modèle une **séquence** des T dernières frames (LSTM), il pourra anticiper en se souvenant de la courbure passée.
>
> 💬 **Question** : visualisez la série temporelle du `road_offset` sur un épisode. Observez-vous une auto-corrélation (la valeur à t prédit bien la valeur à t+1) ? Calculez `pd.Series(df['road_offset']).autocorr(lag=1)`. Qu'en déduisez-vous sur l'utilité d'un LSTM ?

---

## Ressources utiles

- [PyTorch nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- [PyTorch Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [Tutoriel PyTorch — Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- Azencott — *Introduction au Machine Learning* (2e éd.) · Dunod 2022 · Bibliothèque ENI

---

*NRLE822 · Projet fil rouge · Phase 2/4 · EPSI 2025-2026*
