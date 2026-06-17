# Phase 3 — Mémoire temporelle (RNN / LSTM / GRU)
## NRLE822 · Projet fil rouge · CarRacing-v2

> **Phase** 3/4 &nbsp;|&nbsp; **Durée** 4h (S4 — 2h FFP + S5 — 2h classe virtuelle) &nbsp;|&nbsp; **Notebook** `03_LSTM_trajectory.ipynb`  
> **Compétences** CDPEIA 2.5 · 2.6 &nbsp;|&nbsp; **Prérequis** Phases 1 et 2 complétées

---

## Objectif

Remplacer le MLP stateless de la Phase 2 par un **LSTM** capable de mémoriser les frames précédentes pour mieux anticiper les virages. Vous comparerez ensuite LSTM et GRU sur les mêmes métriques.

À la fin de cette phase vous saurez :

- Comprendre les limites des RNN simples et l'intérêt du LSTM
- Transformer une série temporelle en dataset de fenêtres glissantes
- Implémenter `LSTMDriver` et `GRUDriver` avec PyTorch
- Comparer deux architectures de manière rigoureuse

---

## 3.1 Motivation : pourquoi la mémoire ?

### La limite fondamentale du MLP (Phase 2)

Votre `DrivingMLP` prédit `action(t)` uniquement à partir de `observation(t)`. Il ignore tout ce qui s'est passé avant. Conséquence concrète :

```
Step 80 : road_offset = 0.1   → modèle prédit steering = 0.05  (tout droit)
Step 81 : road_offset = 0.2   → modèle prédit steering = 0.10
Step 82 : road_offset = 0.4   → modèle prédit steering = 0.28  (trop tard)
Step 83 : road_offset = 0.7   → modèle prédit steering = 0.55  (toujours en retard)
```

Un conducteur humain, lui, détecte la courbure **croissante** et anticipe dès le step 80. Il a de la **mémoire**.

### L'auto-corrélation du steering

Avant d'implémenter quoi que ce soit, vérifiez empiriquement que la mémoire est utile :

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv('../data/demo_episodes/episode_000.csv')

# Auto-corrélation : dans quelle mesure steering(t) prédit steering(t+lag) ?
lags  = range(1, 31)
corrs = [df['action_steering'].autocorr(lag=lag) for lag in lags]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(df['action_steering'][:300], lw=1, color='steelblue')
axes[0].set(xlabel='Step', ylabel='Steering', title='Série temporelle du steering')
axes[0].grid(alpha=0.3)

axes[1].bar(lags, corrs, color='steelblue', alpha=0.8)
axes[1].set(xlabel='Lag (steps)', ylabel='Corrélation',
            title='Auto-corrélation du steering')
axes[1].axhline(0, color='gray', lw=0.5)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Corrélation à lag=1  : {df["action_steering"].autocorr(1):.3f}')
print(f'Corrélation à lag=5  : {df["action_steering"].autocorr(5):.3f}')
print(f'Corrélation à lag=10 : {df["action_steering"].autocorr(10):.3f}')
```

> 💬 **Question** : que constatez-vous ? Une forte auto-corrélation aux faibles lags justifie-t-elle l'utilisation d'un LSTM ? Expliquez.

---

## 3.2 Rappels théoriques

### Le neurone récurrent (RNN simple)

Un RNN ajoute une **boucle de rétroaction** : l'état caché `hₜ` dépend de l'entrée courante ET de l'état précédent `hₜ₋₁`.

```
hₜ = tanh( Wₕ · hₜ₋₁  +  Wₓ · xₜ  +  b )
         ↑ mémoire        ↑ entrée
```

Si on « déroule » le réseau dans le temps (unrolling) :

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → hₜ → prédiction
               ↑             ↑             ↑
              "mémoire" qui se propage de step en step
```

### Le problème du gradient vanishing

Lors du backward pass sur une séquence longue, les gradients doivent remonter à travers T multiplications successives. Si les poids sont < 1, les gradients **diminuent exponentiellement** → les premiers pas de la séquence ne contribuent plus à l'apprentissage.

```
∂L/∂h₁ = ∂L/∂hₜ × (Wₕ)^(t-1)
                    ↑
          si |Wₕ| < 1 → tend vers 0 pour t grand
          si |Wₕ| > 1 → explose (gradient exploding)
```

C'est le **problème du gradient vanishing / exploding** — la raison pour laquelle les RNN simples peinent à apprendre des dépendances longues.

### La solution : LSTM

Le **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997) introduit deux nouveaux éléments :

- **La cellule mémoire Cₜ** : un « tapis roulant » qui transporte l'information sur de longues distances
- **Trois portes** qui contrôlent ce qui entre, sort et s'efface

```
Porte d'oubli  fₜ = σ( Wf · [hₜ₋₁, xₜ] + bf )   → qu'est-ce qu'on oublie de Cₜ₋₁ ?
Porte d'entrée iₜ = σ( Wi · [hₜ₋₁, xₜ] + bi )   → quelle nouvelle info stocker ?
Candidat       C̃ₜ = tanh( Wc · [hₜ₋₁, xₜ] + bc ) → contenu à potentiellement ajouter
Mise à jour    Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ         → nouvelle cellule mémoire
Porte de sortie oₜ = σ( Wo · [hₜ₋₁, xₜ] + bo )   → quelle partie de Cₜ exposer ?
État caché     hₜ = oₜ ⊙ tanh(Cₜ)                 → sortie du LSTM à ce step
```

> 💡 **Analogie** : le LSTM est comme un rédacteur avec un carnet de notes. À chaque step, il décide quoi effacer du carnet (porte d'oubli), quoi y ajouter (porte d'entrée) et quoi lire pour répondre (porte de sortie).

### La variante GRU

Le **Gated Recurrent Unit** (Cho et al., 2014) simplifie le LSTM en fusionnant certaines portes :

```
Porte de reset  rₜ = σ( Wr · [hₜ₋₁, xₜ] )   → quel passé utiliser pour le candidat ?
Porte update    zₜ = σ( Wz · [hₜ₋₁, xₜ] )   → combien garder de hₜ₋₁ vs nouveau ?
Candidat        h̃ₜ = tanh( W · [rₜ ⊙ hₜ₋₁, xₜ] )
Nouvel état     hₜ = (1−zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

| | LSTM | GRU |
|--|--|--|
| **Portes** | 3 (oubli, entrée, sortie) | 2 (reset, update) |
| **États** | hₜ + Cₜ (cellule mémoire séparée) | hₜ uniquement |
| **Paramètres** | Plus nombreux | Moins nombreux (~25% de moins) |
| **Performance** | Légèrement meilleur sur longues séquences | Souvent comparable, plus rapide |
| **Usage** | Séquences très longues, NLP complexe | Bon point de départ, temps réel |

---

## 3.3 Le dataset séquentiel

### Fenêtres glissantes

Pour exploiter l'historique, on transforme la série temporelle en **fenêtres glissantes** : chaque exemple d'entraînement est une séquence de `T` pas de temps consécutifs, et la cible est l'action au pas `T+1`.

```
Série :    [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, ...]

T=3  →   X = [x₁, x₂, x₃]   y = x₄
          X = [x₂, x₃, x₄]   y = x₅
          X = [x₃, x₄, x₅]   y = x₆
          ...
```

**Paramètre clé** : `seq_len` (= T), la longueur de la fenêtre.

- Trop court (T=1) → équivalent au MLP, pas de mémoire
- Trop long (T=50) → plus de mémoire mais RAM/GPU saturée, gradient vanishing
- Valeur raisonnable : **T = 10** comme point de départ

### `SequenceDataset`

```python
class SequenceDataset(Dataset):
    """
    Transforme une série temporelle en fenêtres glissantes.

    X shape : (N, T, n_features)   — séquences d'historique
    y shape : (N, 3)               — action à prédire au pas T+1
    """
    FEATURES = ['obs_mean_r', 'obs_mean_g', 'obs_mean_b', 'road_offset']
    TARGETS  = ['action_steering', 'action_gas', 'action_brake']

    def __init__(self, df: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len
        X_raw = df[self.FEATURES].values.astype(np.float32)
        y_raw = df[self.TARGETS].values.astype(np.float32)
        X_raw[:, :3] /= 255.0  # Normaliser les couleurs

        # TODO : Construire les fenêtres glissantes
        # seqs, targets = [], []
        # for i in range(len(X_raw) - seq_len):
        #     seqs.append(X_raw[i : i + seq_len])    # (T, n_features)
        #     targets.append(y_raw[i + seq_len])     # action au pas suivant
        #
        # self.X = torch.tensor(np.array(seqs))
        # self.y = torch.tensor(np.array(targets))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

> ⚠️ **Rappel split temporel** : même règle qu'en Phase 2. Le split doit se faire **par épisode entier**, pas ligne par ligne. Les fenêtres ne doivent jamais chevaucher la frontière train/val.

---

## 3.4 Architectures à implémenter

### `LSTMDriver`

```python
class LSTMDriver(nn.Module):
    """
    Modèle LSTM pour la prédiction d'actions séquentielles.

    Entrée : (batch, T, n_features=4)
    Sortie : (batch, 3)

    Architecture :
      Encodeur linéaire  → projette chaque step dans un espace latent
      LSTM (2 couches)   → traite la séquence temporelle
      Dernier état caché → condensé de toute la séquence
      Tête FC            → prédit les 3 actions
    """
    def __init__(self, input_dim=4, hidden_dim=64, n_layers=2,
                 output_dim=3, dropout=0.3):
        super().__init__()

        # Encodeur : projette chaque step features → espace latent 32D
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
        )

        # LSTM : batch_first=True → entrée (batch, seq, features)
        # dropout entre les couches LSTM (seulement si n_layers > 1)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.,
        )

        # Tête de prédiction — on utilise SEULEMENT le dernier état caché
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        # x : (batch, T, 4)
        batch = x.size(0)

        # Encoder chaque step : (batch*T, 4) → (batch*T, 32) → (batch, T, 32)
        enc = self.encoder(x.view(-1, x.size(-1))).view(batch, -1, 32)

        # LSTM : lstm_out (batch, T, hidden), h_n (n_layers, batch, hidden)
        _, (h_n, _) = self.lstm(enc)

        # Dernier état caché de la dernière couche = résumé de toute la séquence
        last_hidden = h_n[-1]   # (batch, hidden_dim)

        return self.head(last_hidden)
```

### `GRUDriver`

```python
class GRUDriver(nn.Module):
    """
    Variante GRU — même interface que LSTMDriver.
    Seulement 2 portes au lieu de 3 → ~25% moins de paramètres.
    """
    def __init__(self, input_dim=4, hidden_dim=64, n_layers=2,
                 output_dim=3, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.gru = nn.GRU(
            input_size=32, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, output_dim),
        )

    def forward(self, x):
        batch = x.size(0)
        enc = self.encoder(x.view(-1, x.size(-1))).view(batch, -1, 32)
        # GRU : pas de cellule mémoire → retourne seulement (output, h_n)
        _, h_n = self.gru(enc)
        return self.head(h_n[-1])
```

> 💬 **Question** : pourquoi utilise-t-on `h_n[-1]` (le dernier état caché) plutôt que `lstm_out[:, -1, :]` (la dernière sortie) ? En pratique ils sont identiques pour 1 couche, mais pas pour 2 couches. Cherchez pourquoi.

---

## 3.5 Travail à réaliser (notebook 03)

### Cellule 1 — Configuration

```python
import numpy as np, pandas as pd, glob, os, sys
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.append('..')
from utils.metrics import (rmse, mae, directional_accuracy,
                           plot_training_curves, compare_models,
                           print_regression_report)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 10  # Longueur de la fenêtre temporelle — à faire varier
```

---

### Cellule 2 — Visualisation de la dépendance temporelle

```python
# TODO : Charger les CSV
# TODO : Tracer la série temporelle du steering (300 premiers steps)
# TODO : Calculer et tracer l'auto-corrélation pour lags 1 à 30
# TODO : Imprimer les corrélations à lag=1, 5, 10
```

---

### Cellule 3 — `SequenceDataset` et DataLoaders

```python
# TODO : Implémenter SequenceDataset (cf. section 3.3)

# Split par épisode — même logique que Phase 2
csv_files = sorted(glob.glob('../data/demo_episodes/episode_*.csv'))
dfs = [pd.read_csv(f) for f in csv_files]
df_ep2 = dfs[2]
mid    = len(df_ep2) // 2

train_ds = SequenceDataset(pd.concat(dfs[:2], ignore_index=True), seq_len=SEQ_LEN)
val_ds   = SequenceDataset(df_ep2.iloc[:mid],  seq_len=SEQ_LEN)
test_ds  = SequenceDataset(df_ep2.iloc[mid:],  seq_len=SEQ_LEN)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print(f'X shape : {train_ds.X.shape}  → (N, T={SEQ_LEN}, features=4)')
print(f'y shape : {train_ds.y.shape}  → (N, 3 actions)')
```

---

### Cellule 4 — Implémentation et vérification des modèles

```python
# TODO : Implémenter LSTMDriver et GRUDriver (cf. section 3.4)

# Vérification des dimensions — ne modifiez pas
for ModelClass, name in [(LSTMDriver, 'LSTM'), (GRUDriver, 'GRU')]:
    m = ModelClass().to(DEVICE)
    dummy = torch.zeros(8, SEQ_LEN, 4).to(DEVICE)
    with torch.no_grad():
        out = m(dummy)
    params = sum(p.numel() for p in m.parameters())
    print(f'{name} | Entrée {dummy.shape} → Sortie {out.shape} | {params:,} paramètres')
    assert out.shape == (8, 3), f'{name} : dimensions incorrectes !'
```

---

### Cellule 5 — Boucle d'entraînement commune

```python
def train_seq_model(model, train_ld, val_ld, n_epochs=25,
                    lr=1e-3, model_name='model', device=DEVICE):
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    history = {'tl': [], 'vl': []}
    best_vl = float('inf')

    for ep in range(1, n_epochs + 1):
        model.train()
        tl = 0.
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            # TODO : forward → loss → zero_grad → backward → step
            tl += loss.item()
        tl /= len(train_ld)

        model.eval()
        vl = 0.
        with torch.no_grad():
            for xb, yb in val_ld:
                # TODO : accumuler vl
                pass
        vl /= len(val_ld)
        sched.step(vl)

        history['tl'].append(tl)
        history['vl'].append(vl)
        if vl < best_vl:
            best_vl = vl
            torch.save(model.state_dict(), f'/tmp/{model_name}_best.pt')
        if ep % 10 == 0:
            print(f'Ep {ep:3d} | Train={tl:.5f} | Val={vl:.5f}')

    model.load_state_dict(torch.load(f'/tmp/{model_name}_best.pt'))
    return history


# Entraîner les deux modèles
print('=== LSTMDriver ===')
lstm_model = LSTMDriver().to(DEVICE)
hist_lstm  = train_seq_model(lstm_model, train_ld, val_ld, model_name='lstm')

print('\n=== GRUDriver ===')
gru_model  = GRUDriver().to(DEVICE)
hist_gru   = train_seq_model(gru_model,  train_ld, val_ld, model_name='gru')
```

---

### Cellule 6 — Comparaison LSTM vs GRU

```python
def evaluate_seq(model, loader, device=DEVICE):
    """Retourne y_true et y_pred sur tout le loader."""
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(preds)


y_true_lstm, y_pred_lstm = evaluate_seq(lstm_model, test_ld)
y_true_gru,  y_pred_gru  = evaluate_seq(gru_model,  test_ld)

# Rapport détaillé pour le steering
print('─── LSTM ───')
print_regression_report(y_true_lstm[:, 0], y_pred_lstm[:, 0], label='Steering')

print('─── GRU  ───')
print_regression_report(y_true_gru[:,  0], y_pred_gru[:,  0], label='Steering')

# Graphe de comparaison
results = {
    'LSTM': {
        'rmse':     rmse(y_true_lstm[:, 0], y_pred_lstm[:, 0]),
        'dir_acc':  directional_accuracy(y_true_lstm[:, 0], y_pred_lstm[:, 0]),
        'val_loss': min(hist_lstm['vl']),
    },
    'GRU': {
        'rmse':     rmse(y_true_gru[:, 0], y_pred_gru[:, 0]),
        'dir_acc':  directional_accuracy(y_true_gru[:, 0], y_pred_gru[:, 0]),
        'val_loss': min(hist_gru['vl']),
    },
}
compare_models(results)
```

---

### Cellule 7 — Comparaison avec le MLP de la Phase 2

```python
# TODO : Recharger votre DrivingMLP entraîné en Phase 2
# (ou ré-entraîner rapidement sur les mêmes données)
# Comparez RMSE et précision directionnelle : MLP vs LSTM vs GRU
# Ajoutez le MLP dans le dictionnaire results ci-dessus et re-tracer compare_models
```

> 💬 **Question** : le LSTM améliore-t-il significativement le MLP ? Sur quel type de situations (virages courts, longs, enchaînés) l'amélioration est-elle la plus visible ?

---

### Cellule 8 — Expérimentation : impact de `seq_len`

```python
# TODO : Entraîner LSTMDriver avec seq_len ∈ {5, 10, 20}
# Pour chaque valeur, stocker le RMSE test sur le steering
# Tracer un graphe seq_len vs RMSE

seq_lens = [5, 10, 20]
rmse_by_seq = {}

for T in seq_lens:
    # Recréer les datasets avec ce T
    train_ds_t = SequenceDataset(pd.concat(dfs[:2], ignore_index=True), seq_len=T)
    val_ds_t   = SequenceDataset(df_ep2.iloc[:mid], seq_len=T)
    test_ds_t  = SequenceDataset(df_ep2.iloc[mid:], seq_len=T)
    # TODO : Créer les loaders, entraîner, évaluer, stocker le RMSE
```

> 💬 **Question** : au-delà de quelle valeur de `seq_len` les performances commencent-elles à se dégrader ? Formulez une hypothèse.

---

## 3.6 Points de vigilance

### `batch_first=True` — indispensable

PyTorch attend par défaut les tenseurs au format `(seq, batch, features)`. Avec `batch_first=True`, on utilise `(batch, seq, features)` — beaucoup plus intuitif. **Ne pas l'oublier sinon les dimensions seront silencieusement échangées.**

```python
# ❌ Sans batch_first : entrée attendue (T, batch, features)
lstm = nn.LSTM(input_size=32, hidden_size=64)

# ✅ Avec batch_first : entrée attendue (batch, T, features)
lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
```

### `dropout` dans LSTM/GRU

Le paramètre `dropout` du LSTM/GRU s'applique **entre les couches**, pas sur la sortie finale. Il est ignoré si `num_layers=1` — dans ce cas, ajoutez un `nn.Dropout` manuellement dans la tête.

### Gradient clipping

Les RNN peuvent souffrir d'explosion de gradient. Si votre loss explose soudainement, ajoutez :

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# À placer entre loss.backward() et optimizer.step()
```

---

## 3.7 Questions de réflexion

À rédiger dans des **cellules Markdown** du notebook :

1. L'auto-corrélation que vous avez calculée en cellule 2 justifie-t-elle l'utilisation d'un LSTM ? Montrez le lien entre la corrélation à lag=T et le choix de `seq_len`.

2. Comparez le nombre de paramètres de LSTMDriver et GRUDriver. Le GRU est-il plus performant par paramètre (RMSE / nombre de paramètres) ?

3. La boucle d'entraînement LSTM est-elle plus lente que celle du MLP ? Mesurez le temps avec `time.time()` et commentez le compromis performance / vitesse.

4. *(Optionnel)* Que se passe-t-il si vous utilisez `lstm_out[:, -1, :]` (dernière sortie) au lieu de `h_n[-1]` (dernier état caché) pour un LSTM à 2 couches ? Testez et expliquez la différence.

---

## 3.8 Limitation et transition vers la Phase 4

> ⚠️ **Ce que la Phase 3 ne peut pas faire**
>
> Votre LSTM imite un pilote humain — mais ce pilote lui-même n'est pas parfait. Si les données d'entraînement contiennent des virages ratés ou des freinages tardifs, le LSTM les reproduira fidèlement.
>
> Plus fondamentalement : l'imitation learning est **borné par la qualité du professeur**. Pour dépasser les performances humaines — ou s'adapter à des conditions jamais vues dans les données — il faut apprendre par **essai-erreur directement dans l'environnement**.
>
> C'est l'objet de la Phase 4 : l'agent DQN découvrira lui-même la politique optimale en maximisant la récompense de CarRacing-v2, sans jamais avoir besoin de données étiquetées.
>
> 💬 **Question** : imaginez un virage que votre LSTM n'a jamais vu dans les données d'entraînement. Comment se comportera-t-il ? Un agent DQN entraîné suffisamment longtemps serait-il plus robuste ? Pourquoi ?

---

## Ressources utiles

- [Understanding LSTMs — Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) ⭐ incontournable
- [Visualisation interactive LSTM — Distill](https://distill.pub/2019/memorization-in-rnns/)
- [PyTorch nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- Sutton & Barto — *Reinforcement Learning: An Introduction* · MIT Press 2018 (libre de droit)

---

*NRLE822 · Projet fil rouge · Phase 3/4 · EPSI 2025-2026*
