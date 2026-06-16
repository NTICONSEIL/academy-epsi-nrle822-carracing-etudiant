# 🏎️ Projet fil rouge — Conduite Autonome avec CarRacing-v2
### NRLE822 · Neural Networks and Reinforcement Learning · Semestre 8

---

## Contexte

Vous êtes mandaté·e par une startup de mobilité autonome pour concevoir un pipeline IA capable de piloter une voiture virtuelle sur le circuit **CarRacing-v2**.

Le projet se déroule en **4 phases progressives**, chacune correspondant à un notebook Jupyter à compléter et à commenter. L'ensemble constitue votre livrable pour la **MSPR TPRE822**.

> **Environnement :** [CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/) — Gymnasium  
> **Frameworks :** PyTorch · NumPy · Pandas · Matplotlib  
> **Durée totale :** 14h (cours + travail personnel)

---

## Structure du repo

```
carracing-etudiant/
│
├── README.md                    ← Ce fichier — lisez-le en premier
│
├── notebooks/
│   ├── 01_CNN_perception.ipynb         ← Phase 1 (3h)
│   ├── 02_Supervised_imitation.ipynb   ← Phase 2 (4h)
│   ├── 03_LSTM_trajectory.ipynb        ← Phase 3 (4h)
│   └── 04_DQN_agent.ipynb              ← Phase 4 (3h)
│
├── data/
│   └── demo_episodes/           ← Données de conduite pré-générées (CSV)
│       ├── episode_000.csv
│       ├── episode_001.csv
│       ├── episode_002.csv
│       └── metadata.csv
│
└── utils/
    ├── env_wrappers.py          ← Wrappers Gymnasium (fournis)
    ├── metrics.py               ← Fonctions d'évaluation (fournies)
    ├── replay_buffer.py         ← Replay buffer DQN (fourni)
    └── viz.py                   ← Visualisations (fournies)
```

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/VOTRE_ORG/nrle822-carracing-etudiant.git
cd nrle822-carracing-etudiant
```

### 2. Créer l'environnement Python

```bash
conda create -n carracing python=3.10 -y
conda activate carracing
```

### 3. Installer les dépendances

```bash
# Linux uniquement — dépendances système pour Box2D
sudo apt-get install -y swig libgl1-mesa-glx

# Packages Python
pip install swig
pip install gymnasium[box2d]==0.29.1
pip install torch==2.2.0 torchvision==0.17.0
pip install numpy pandas matplotlib seaborn jupyter ipywidgets tqdm pillow pygame
```

### 4. Vérifier l'installation

```bash
python -c "
import gymnasium as gym, torch
env = gym.make('CarRacing-v2', render_mode=None)
obs, _ = env.reset()
print(f'✅ Gym OK — obs: {obs.shape}')
print(f'✅ PyTorch OK — {\"GPU\" if torch.cuda.is_available() else \"CPU\"}')
env.close()
"
```

Si vous obtenez une erreur `Box2D not found`, consultez la section [Erreurs fréquentes](#erreurs-fréquentes) en bas de ce fichier.

---

## Vue d'ensemble du projet

### L'environnement

À chaque pas de temps, la voiture reçoit une **observation** (image RGB 96×96) et doit produire une **action** (direction, accélération, freinage).

```
┌─────────────────────────────────────────────────────┐
│                  CarRacing-v2                        │
│                                                     │
│  Observation : image RGB 96×96×3                    │
│  Action      : [steering, gas, brake]               │
│               steering ∈ [-1, 1]   (gauche/droite) │
│               gas      ∈ [0, 1]    (accélérer)     │
│               brake    ∈ [0, 1]    (freiner)        │
│  Récompense  : +score par tuile de route visitée    │
│               -0.1 par frame (pénalité de temps)    │
└─────────────────────────────────────────────────────┘
```

### Les 4 phases

| Phase | Notebook | Approche | Question centrale |
|-------|----------|----------|-------------------|
| **1 — Perception** | `01_CNN_perception` | CNN supervisé | *Comment un réseau voit-il la route ?* |
| **2 — Imitation** | `02_Supervised_imitation` | Régression supervisée | *Comment imiter un pilote humain ?* |
| **3 — Mémoire** | `03_LSTM_trajectory` | RNN / LSTM | *Comment anticiper en se souvenant du passé ?* |
| **4 — Agent** | `04_DQN_agent` | Q-Learning / DQN | *Comment apprendre par essai-erreur ?* |

Chaque phase **réutilise et enrichit** la précédente. Le CNN de la Phase 1 devient l'encodeur visuel des phases suivantes.

---

## Phase 1 — Perception de l'environnement

**Durée :** 3h · Séances S1 (2h FFP) + S2 (1h classe virtuelle)

### Objectif

Construire un CNN capable de **classifier le type de route** visible dans une frame : ligne droite, virage à gauche, virage à droite.

### Ce que vous allez implémenter

```python
class PerceptionCNN(nn.Module):
    # 3 couches convolutionnelles + BatchNorm
    # Tête de classification avec Dropout
    # Entrée  : (batch, 3, 96, 96)
    # Sortie  : (batch, 3)  — logits pour 3 classes
```

### Concepts clés à maîtriser

- **Neurone artificiel** : poids, biais, fonction d'activation
- **Convolution** : filtre glissant, détection de patterns locaux
- **BatchNorm** : stabilise l'entraînement en normalisant les activations
- **Dropout** : régularisation par désactivation aléatoire de neurones
- **Feature maps** : visualisation de ce que "voit" chaque filtre

### Jalons du notebook `01_CNN_perception.ipynb`

- [ ] Cellule 1 — Fixer les seeds et vérifier le device
- [ ] Cellule 2 — Explorer l'environnement et visualiser 8 frames
- [ ] Cellule 3 — Tracer les 4 fonctions d'activation (ReLU, Sigmoid, Tanh, Leaky ReLU)
- [ ] Cellule 4 — Implémenter `PerceptionCNN` et vérifier les dimensions
- [ ] Cellule 5 — Collecter ~1 200 frames et créer les DataLoaders (split 70/15/15)
- [ ] Cellule 6 — Écrire la boucle d'entraînement
- [ ] Cellule 7 — Tracer et commenter les courbes de loss et d'accuracy
- [ ] Cellule 8 — Visualiser les feature maps de la couche Conv1

### Questions de réflexion (à rédiger en Markdown dans le notebook)

1. Pourquoi un CNN est-il plus adapté qu'un MLP pour traiter des images ?
2. Qu'observez-vous dans les feature maps ? Que détecte chaque filtre ?
3. Vos courbes montrent-elles du surapprentissage ? Comment le diagnostiquer et le corriger ?

---

## Phase 2 — Imitation d'un pilote

**Durée :** 4h · Séance S3 (4h FFP)

### Objectif

Entraîner un modèle à **prédire les actions de conduite** (steering, gas, brake) à partir des observations, en imitant un pilote rule-based.

### Ce que vous allez implémenter

```python
class DrivingMLP(nn.Module):
    # MLP 4 couches avec Dropout
    # Entrée  : features résumées [R_moy, G_moy, B_moy, road_offset]
    # Sortie  : [steering, gas, brake]  — régression multi-sortie
```

### Concepts clés à maîtriser

- **Rétropropagation** : calcul des gradients par la règle de la chaîne
- **MSE Loss** : perte adaptée à la régression (vs CrossEntropy pour la classification)
- **Adam** : optimiseur adaptatif — comprendre le rôle du learning rate
- **Split temporel** : pourquoi mélanger des frames séquentielles crée du data leakage
- **Métriques de régression** : RMSE, MAE, précision directionnelle

### ⚠️ Point critique : le split temporel

Pour des données de conduite (séries temporelles), le split doit se faire **par épisode entier**, pas ligne par ligne :

```python
# ❌ FAUX — data leakage : des frames consécutives se retrouvent dans train ET val
df_shuffled = df.sample(frac=1)
train, val = df_shuffled[:800], df_shuffled[800:]

# ✅ CORRECT — on garde les épisodes intacts
train_df = episodes[0] + episodes[1]   # Épisodes 0 et 1 → train
val_df   = episodes[2][:mid]           # Moitié de l'épisode 2 → val
test_df  = episodes[2][mid:]           # Reste → test
```

### Jalons du notebook `02_Supervised_imitation.ipynb`

- [ ] Charger les CSV et explorer les distributions des actions
- [ ] Implémenter le split train/val/test par épisode (justifier dans une cellule Markdown)
- [ ] Créer `DrivingDataset` héritant de `torch.utils.data.Dataset`
- [ ] Implémenter `DrivingMLP` et vérifier les dimensions
- [ ] Boucle d'entraînement avec MSELoss + Adam + scheduler
- [ ] Calculer RMSE, MAE et précision directionnelle sur le test set
- [ ] Comparer avec une baseline naïve (prédire la moyenne d'entraînement)
- [ ] Tester au moins 2 valeurs de learning rate et commenter l'impact

### Questions de réflexion

1. Que se passe-t-il si vous mélangez les lignes avant de splitter ? Montrez-le empiriquement.
2. Le RMSE seul suffit-il pour évaluer un pilote ? Pourquoi proposer la précision directionnelle ?
3. Quelle limitation fondamentale a un modèle qui prédit chaque frame indépendamment ?

---

## Phase 3 — Mémoire temporelle

**Durée :** 4h · Séances S4 (2h FFP) + S5 (2h classe virtuelle)

### Objectif

Remplacer le MLP stateless par un **LSTM** capable de mémoriser les frames précédentes pour mieux anticiper les virages.

### Ce que vous allez implémenter

```python
class LSTMDriver(nn.Module):
    # Encodeur linéaire → LSTM (2 couches) → tête de régression
    # Entrée  : (batch, T, 4)  — séquence de T pas de temps
    # Sortie  : (batch, 3)     — action au pas T+1

class GRUDriver(nn.Module):
    # Variante simplifiée du LSTM (2 portes au lieu de 3)
    # Mêmes dimensions d'entrée/sortie
```

### Concepts clés à maîtriser

- **RNN** : état caché hₜ = f(hₜ₋₁, xₜ) — mémoire à court terme
- **Gradient vanishing** : pourquoi les RNN simples oublient les longues dépendances
- **LSTM** : portes d'entrée, d'oubli et de sortie — cellule mémoire Cₜ
- **GRU** : variante à 2 portes, plus légère que le LSTM
- **Fenêtres glissantes** : transformer une série temporelle en dataset supervisé

### Dataset séquentiel

```
Série temporelle :   [x₁, x₂, x₃, x₄, x₅, x₆, x₇, ...]

Fenêtres (T=3) :     X = [x₁, x₂, x₃]  →  y = x₄
                     X = [x₂, x₃, x₄]  →  y = x₅
                     X = [x₃, x₄, x₅]  →  y = x₆
```

### Jalons du notebook `03_LSTM_trajectory.ipynb`

- [ ] Visualiser l'auto-corrélation du steering (justifier l'utilité du LSTM)
- [ ] Implémenter `SequenceDataset` avec fenêtres glissantes (paramètre `seq_len`)
- [ ] Implémenter `LSTMDriver` — attention à `batch_first=True`
- [ ] Implémenter `GRUDriver` comme variante
- [ ] Entraîner les deux modèles et comparer leurs courbes de loss
- [ ] Comparer LSTM vs GRU sur RMSE, MAE et précision directionnelle
- [ ] Expérimenter avec `seq_len` ∈ {5, 10, 20} et analyser l'impact

### Questions de réflexion

1. L'auto-corrélation du steering justifie-t-elle l'usage d'un LSTM ? Montrez-le avec un graphe.
2. Quelle différence de performance observez-vous entre LSTM et GRU ? L'expliquer.
3. Au-delà de quelle longueur de séquence les performances se dégradent-elles ? Pourquoi ?

---

## Phase 4 — Agent autonome

**Durée :** 3h · Séances S6 (2h FFP) + S7 (1h classe virtuelle)

### Objectif

Implémenter un agent **DQN** qui apprend à conduire par essai-erreur, sans données supervisées, en maximisant la récompense de l'environnement.

### Ce que vous allez implémenter

```python
class DQN(nn.Module):
    # CNN + tête FC — prédit Q(s, a) pour chaque action discrète
    # Entrée  : (batch, 1, 42, 42)  — frame grayscale
    # Sortie  : (batch, 5)          — Q-valeur pour 5 actions

class DQNAgent:
    # policy_net + target_net + ReplayBuffer
    # Méthodes : select_action(), store(), train_step()
    #            update_epsilon(), update_target()
```

### Concepts clés à maîtriser

- **MDP** : état S, action A, récompense R, politique π
- **Q-Learning** : Q(s,a) = r + γ · max_{a'} Q(s', a')
- **Équation de Bellman** : définit la cible d'entraînement
- **ε-greedy** : balance exploration (actions aléatoires) et exploitation
- **Experience replay** : stocker et ré-échantillonner les transitions
- **Target network** : réseau figé pour stabiliser les cibles Bellman

### L'équation de Bellman

```
Q(sₜ, aₜ)  ←  rₜ  +  γ · max_{a'} Q_target(sₜ₊₁, a')
     ↑              ↑         ↑
 valeur courante  récompense  valeur future actualisée
 (policy net)     immédiate   (target net — figé)
```

`γ` (gamma) est le facteur d'actualisation : γ = 0.95 signifie qu'une récompense dans le futur vaut 95% de sa valeur immédiate.

### Jalons du notebook `04_DQN_agent.ipynb`

- [ ] Créer l'env discrétisé : `make_env(grayscale=True, resize=(42,42), discrete=True)`
- [ ] Implémenter `DQN` (CNN + FC, sans activation sur la dernière couche)
- [ ] Implémenter `select_action()` avec stratégie ε-greedy
- [ ] Implémenter `train_step()` avec l'équation de Bellman et Huber Loss
- [ ] Coder la boucle d'entraînement complète (50 épisodes minimum)
- [ ] Tracer la courbe de récompense avec moyenne mobile
- [ ] Tester γ ∈ {0.90, 0.95, 0.99} et analyser l'impact sur le comportement de l'agent

### Questions de réflexion

1. Pourquoi utilise-t-on deux réseaux (policy et target) plutôt qu'un seul ?
2. Que se passe-t-il si ε ne décroît pas (reste à 1.0) ? Et s'il tombe à 0 trop vite ?
3. Quelle est la différence fondamentale entre l'apprentissage par renforcement et l'imitation learning de la Phase 2 ?

---

## Livrable final (MSPR)

### Ce que vous rendez

Un **unique Notebook Jupyter** exécutable de bout en bout, contenant votre solution complète documentée.

### Checklist de rendu

- [ ] Toutes les cellules s'exécutent sans erreur (kernel restart → run all)
- [ ] Les seeds sont fixés (`SEED = 42` partout)
- [ ] Chaque section de code est précédée d'une cellule Markdown explicative
- [ ] Le split train/val/test est justifié
- [ ] Au moins une courbe de loss commentée par phase
- [ ] Au moins deux modèles comparés (ex. LSTM vs GRU, ou DQN avec γ différents)
- [ ] Une conclusion rédigée : forces, limites, améliorations proposées

### Grille d'évaluation (20 pts)

| Critère | Points |
|---------|--------|
| Architecture et conception du modèle | 6 pts |
| Procédure d'entraînement et split | 6 pts |
| Analyse des performances | 5 pts |
| Qualité et lisibilité du notebook | 3 pts |

---

## Erreurs fréquentes

### `Box2D not found`
```bash
pip install swig
pip install gymnasium[box2d]
```

### `CUDA out of memory`
```python
# Réduire la taille du batch
DataLoader(dataset, batch_size=16, ...)
# Ou forcer le CPU
DEVICE = torch.device('cpu')
```

### Loss NaN pendant l'entraînement
```python
# Ajouter le gradient clipping avant optimizer.step()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Le kernel Jupyter meurt silencieusement
Cause probable : mémoire insuffisante lors de la collecte de frames.  
Solution : réduire `n_frames` ou utiliser `resize=(42, 42)`.

---

## Ressources

| Ressource | Lien |
|-----------|------|
| Documentation Gymnasium | https://gymnasium.farama.org |
| PyTorch tutorials | https://pytorch.org/tutorials |
| Understanding LSTMs | https://colah.github.io/posts/2015-08-Understanding-LSTMs |
| Spinning Up in Deep RL | https://spinningup.openai.com |
| Azencott — Introduction au ML | Bibliothèque ENI |
| Sutton & Barto — RL: An Introduction | http://incompleteideas.net/book/the-book-2nd.html |

---

*Module NRLE822 · Mastère SIN — Parcours Expert en Intelligence Artificielle · EPSI 2025-2026*
