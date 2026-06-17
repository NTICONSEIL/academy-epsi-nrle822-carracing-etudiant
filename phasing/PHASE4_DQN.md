# Phase 4 — Agent autonome (Q-Learning / DQN)
## NRLE822 · Projet fil rouge · CarRacing-v2

> **Phase** 4/4 &nbsp;|&nbsp; **Durée** 3h (S6 — 2h FFP + S7 — 1h classe virtuelle) &nbsp;|&nbsp; **Notebook** `04_DQN_agent.ipynb`  
> **Compétences** CDPEIA 2.5 · 2.6 · 2.7 &nbsp;|&nbsp; **Prérequis** Phases 1, 2 et 3 complétées

---

## Objectif

Implémenter un agent **DQN (Deep Q-Network)** qui apprend à conduire par essai-erreur directement dans l'environnement CarRacing-v2, sans données étiquetées. L'agent maximise une récompense cumulative en découvrant lui-même la politique de conduite optimale.

À la fin de cette phase vous saurez :

- Formaliser un problème de conduite comme un processus de décision markovien (MDP)
- Comprendre Q-Learning et l'équation de Bellman
- Implémenter un DQN avec experience replay et target network
- Analyser la courbe de récompense et diagnostiquer l'apprentissage par renforcement

---

## 4.1 Du supervisé au renforcement : un changement de paradigme

### Ce que les Phases 2 et 3 ne peuvent pas faire

Les phases précédentes reposaient sur l'**imitation learning** : votre modèle copiait un pilote rule-based. Deux limitations fondamentales :

- **Borné par le professeur** : le modèle ne peut pas dépasser les performances des données d'entraînement
- **Pas d'adaptation** : face à une situation jamais vue dans le dataset, le modèle extrapole aveuglément

### L'apprentissage par renforcement

L'agent n'a plus besoin de données étiquetées. Il apprend en **interagissant directement avec l'environnement** :

```
         ┌──────────────────────────────────┐
         │                                  │
    sₜ   ▼          aₜ                     │
  ──────► AGENT ──────────► ENVIRONNEMENT   │
         │                      │           │
         │          rₜ, sₜ₊₁   │           │
         └──────────────────────┘           │
              boucle d'interaction          │
         └──────────────────────────────────┘
```

À chaque pas de temps :
1. L'agent observe l'état `sₜ` (frame RGB)
2. Il choisit une action `aₜ` (gauche, droite, accélérer…)
3. L'environnement retourne une récompense `rₜ` et le nouvel état `sₜ₊₁`
4. L'agent met à jour sa politique pour maximiser les récompenses futures

---

## 4.2 Rappels théoriques

### Le formalisme MDP

Un **Processus de Décision Markovien** (MDP) se définit par 4 composantes :

| Composante | CarRacing-v2 | Notation |
|------------|-------------|----------|
| **État** | Frame RGB 42×42 (niveaux de gris) | S |
| **Action** | {tout droit, gauche, droite, accélérer, freiner} | A |
| **Récompense** | +score par tuile · −0.1/frame | R |
| **Transition** | Physique du simulateur | P(s'\|s,a) |

**Propriété de Markov** : l'état futur ne dépend que de l'état présent et de l'action, pas de l'historique. C'est pour ça qu'on parle de *processus* markovien.

### La fonction Q

La **fonction Q** (ou fonction de valeur-action) mesure l'espérance de récompense cumulée en prenant l'action `a` dans l'état `s`, puis en suivant la politique optimale :

```
Q*(s, a) = E[ rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ... ]
                ↑
         récompenses futures actualisées
```

`γ` (gamma) est le **facteur d'actualisation** ∈ [0, 1] :
- `γ = 0` : agent myope — seule la récompense immédiate compte
- `γ = 1` : agent prévoyant — toutes les récompenses futures ont le même poids
- `γ = 0.95` : valeur typique — une récompense dans 20 steps vaut 0.95²⁰ ≈ 36% de sa valeur

### L'équation de Bellman

La propriété récursive de Q* : la valeur de (s, a) est égale à la récompense immédiate plus la valeur actualisée du meilleur état suivant.

```
Q*(sₜ, aₜ) = rₜ  +  γ · max_{a'} Q*(sₜ₊₁, a')
               ↑              ↑
         récompense     meilleure valeur
         immédiate      au pas suivant
```

C'est cette équation qui sert de **cible d'entraînement** pour le réseau de neurones.

### De Q-Learning à DQN

Le Q-Learning classique stocke Q(s, a) dans une table — impossible quand l'espace d'états est une image 42×42. Le **DQN** (Mnih et al., 2015) remplace cette table par un **réseau de neurones** :

```
Frame RGB  →  CNN  →  FC  →  [Q(s, a₀), Q(s, a₁), Q(s, a₂), Q(s, a₃), Q(s, a₄)]
                               ↑ tout    ↑ gauche   ↑ droite  ↑ accel.  ↑ frein
```

Le réseau prédit la Q-valeur pour **toutes les actions simultanément** en un seul forward pass.

### La stratégie ε-greedy

L'agent doit trouver l'équilibre entre :
- **Exploration** : tester des actions aléatoires pour découvrir de nouvelles stratégies
- **Exploitation** : choisir l'action avec la meilleure Q-valeur connue

La stratégie ε-greedy résout ce dilemme :

```python
if random() < epsilon:
    action = random_action()      # Exploration
else:
    action = argmax Q(s, a)       # Exploitation
```

ε décroît au fil de l'entraînement : l'agent explore beaucoup au début, puis exploite de plus en plus.

```
ε = 1.0  →  0.05
     │         │
     │  decay  │
     └─────────┘
  épisode 1   épisode 300
```

### Experience Replay

Problème : les transitions consécutives `(sₜ, aₜ, rₜ, sₜ₊₁)` sont fortement corrélées → l'entraînement est instable.

Solution : stocker les transitions dans un **buffer** et en tirer des mini-batches **aléatoires** à chaque étape d'entraînement.

```python
# Stocker
buffer.push(state, action, reward, next_state, done)

# Entraîner sur un batch aléatoire
states, actions, rewards, next_states, dones = buffer.sample(batch_size=64)
```

Avantages :
- Brise les corrélations temporelles
- Chaque transition peut être utilisée plusieurs fois
- Stabilise la descente de gradient

### Target Network

Problème : si on utilise le même réseau pour calculer la prédiction ET la cible, les deux bougent en même temps → instabilité.

Solution : utiliser **deux réseaux** :

| Réseau | Rôle | Mise à jour |
|--------|------|-------------|
| **Policy net** | Prédit Q(s, a) — sert à choisir les actions | À chaque step |
| **Target net** | Calcule les cibles Bellman | Copie du policy net toutes les N étapes |

```python
# Cible Bellman avec le target net (figé)
with torch.no_grad():
    q_next   = target_net(next_states).max(dim=1)[0]
    q_target = rewards + gamma * q_next * (1 - dones)

# Prédiction avec le policy net
q_current = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

# Loss
loss = F.smooth_l1_loss(q_current, q_target)
```

---

## 4.3 L'environnement discrétisé

DQN fonctionne avec un espace d'action **discret**. Le wrapper `DiscretizeActionWrapper` fourni dans `utils/env_wrappers.py` transforme l'espace continu de CarRacing-v2 en 5 actions :

| Index | Label | `[steering, gas, brake]` |
|-------|-------|--------------------------|
| 0 | Tout droit | `[0.0, 0.5, 0.0]` |
| 1 | Gauche léger | `[-0.5, 0.5, 0.0]` |
| 2 | Droite léger | `[+0.5, 0.5, 0.0]` |
| 3 | Accélérer | `[0.0, 1.0, 0.0]` |
| 4 | Freiner | `[0.0, 0.0, 0.8]` |

Pour créer l'environnement complet :

```python
from utils.env_wrappers import make_env

env = make_env(
    render_mode=None,   # Pas de fenêtre graphique → x10 plus rapide
    grayscale=True,     # Niveaux de gris → 1 canal au lieu de 3
    resize=(42, 42),    # Réduire la résolution → moins de paramètres CNN
    normalize=True,     # Pixels dans [0, 1]
    discrete=True,      # Espace d'action discret (5 actions)
)

print('Observation :', env.observation_space)  # Box(42, 42, 1)
print('Action      :', env.action_space)       # Discrete(5)
```

---

## 4.4 Architecture à implémenter

### `DQN` — Le réseau Q

```python
class DQN(nn.Module):
    """
    Deep Q-Network pour CarRacing-v2 discrétisé.

    Entrée  : (batch, 1, 42, 42)  — frame grayscale normalisée
    Sortie  : (batch, n_actions)  — Q-valeur pour chaque action

    Pas d'activation sur la dernière couche :
    les Q-valeurs peuvent être négatives.
    """
    def __init__(self, n_actions: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),   # → (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # → (32, 9, 9)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),  # → (32, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),  # Pas d'activation finale
        )

    def forward(self, x):
        # Gymnasium retourne (batch, H, W, C) → PyTorch attend (batch, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.conv(x))
```

### `DQNAgent` — L'agent complet

```python
class DQNAgent:
    """
    Agent DQN avec experience replay et target network.

    Paramètres clés
    ---------------
    gamma        : facteur d'actualisation (0.95 par défaut)
    eps_start    : ε initial — exploration maximale
    eps_end      : ε final  — exploitation quasi-totale
    eps_decay    : taux de décroissance de ε par épisode
    buffer_size  : capacité du replay buffer
    batch_size   : taille du mini-batch d'entraînement
    target_update: fréquence de synchronisation policy → target (en épisodes)
    """
    def __init__(
        self, n_actions=5, lr=1e-4, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay=0.995,
        buffer_size=5_000, batch_size=64, target_update=10
    ):
        # TODO : Instancier policy_net et target_net (même architecture DQN)
        # TODO : Copier les poids policy → target et mettre target en eval()
        # TODO : Instancier optimizer (Adam) et buffer (ReplayBuffer)
        pass

    def select_action(self, obs: np.ndarray) -> int:
        """Stratégie ε-greedy."""
        # TODO : Si random() < eps → action aléatoire
        # TODO : Sinon → argmax Q(obs, a) via policy_net
        pass

    def store(self, obs, action, reward, next_obs, done):
        """Stocke une transition dans le replay buffer."""
        # TODO : buffer.push(...)
        pass

    def train_step(self) -> float:
        """
        Un pas d'optimisation sur un mini-batch.
        Retourne la loss, ou 0.0 si le buffer n'est pas encore assez plein.
        """
        # TODO : Vérifier buffer.is_ready(batch_size)
        # TODO : Échantillonner un batch
        # TODO : Calculer q_current avec policy_net
        # TODO : Calculer q_target avec target_net (torch.no_grad())
        # TODO : Calculer la Huber Loss (F.smooth_l1_loss)
        # TODO : Backpropagation + gradient clipping (max_norm=10)
        pass

    def update_epsilon(self):
        """Décroît ε après chaque épisode."""
        # TODO : self.eps = max(eps_end, eps * eps_decay)
        pass

    def update_target(self, episode: int):
        """Synchronise policy_net → target_net toutes les target_update épisodes."""
        # TODO : if episode % target_update == 0 → copier les poids
        pass
```

---

## 4.5 Travail à réaliser (notebook 04)

### Cellule 1 — Configuration

```python
import numpy as np, sys, os, random, time
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
sys.path.append('..')
from utils.env_wrappers import make_env, DiscretizeActionWrapper
from utils.replay_buffer import ReplayBuffer
from utils.metrics import plot_reward_curve

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
```

---

### Cellule 2 — Environnement et vérification

```python
# TODO : Créer l'environnement avec make_env(grayscale=True, resize=(42,42), discrete=True)

# Afficher les espaces
print('Observation :', env.observation_space)
print('Action      :', env.action_space)
print('Actions disponibles :')
for i, a in enumerate(DiscretizeActionWrapper.ACTIONS):
    labels = ['tout droit', 'gauche léger', 'droite léger', 'accélérer', 'freiner']
    print(f'  {i} : {labels[i]:15s}  [steering={a[0]:+.1f}, gas={a[1]:.1f}, brake={a[2]:.1f}]')

# Vérifier une transition
obs, _ = env.reset(seed=SEED)
next_obs, reward, terminated, truncated, _ = env.step(0)
print(f'\nObs shape  : {obs.shape}   dtype={obs.dtype}')
print(f'Reward     : {reward:.3f}')
```

---

### Cellule 3 — Réseau DQN et vérification des dimensions

```python
# TODO : Implémenter la classe DQN (cf. section 4.4)

# Vérification obligatoire
net = DQN(n_actions=5).to(DEVICE)
dummy = torch.zeros(4, 42, 42, 1).to(DEVICE)   # 4 frames grayscale
with torch.no_grad():
    q_vals = net(dummy)
print(f'Entrée  : {dummy.shape}')
print(f'Sortie  : {q_vals.shape}')   # Attendu : torch.Size([4, 5])
assert q_vals.shape == (4, 5), 'Dimensions incorrectes !'

total = sum(p.numel() for p in net.parameters())
print(f'Paramètres : {total:,}')
```

---

### Cellule 4 — Agent DQN

```python
# TODO : Implémenter la classe DQNAgent (cf. section 4.4)

# Vérification
agent = DQNAgent(n_actions=5)
obs, _ = env.reset(seed=SEED)
action = agent.select_action(obs)
print(f'Action choisie : {action}  (doit être un entier entre 0 et 4)')
assert isinstance(action, int) and 0 <= action < 5
print('✅ DQNAgent opérationnel')
```

---

### Cellule 5 — Visualisation de la décroissance ε

```python
# Visualiser la décroissance de epsilon avant de lancer l'entraînement
eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
epsilons = []
eps = eps_start
for ep in range(400):
    eps = max(eps_end, eps * eps_decay)
    epsilons.append(eps)

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(epsilons, color='coral', lw=2)
ax.axhline(eps_end, color='gray', ls='--', label=f'ε_min = {eps_end}')
ax.fill_between(range(400), epsilons, eps_end, alpha=0.1, color='coral',
                label='zone exploration')
ax.set_xlabel('Épisode')
ax.set_ylabel('ε (epsilon)')
ax.set_title('Décroissance de ε : exploration → exploitation')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'ε après 50  épisodes : {epsilons[49]:.3f}')
print(f'ε après 100 épisodes : {epsilons[99]:.3f}')
print(f'ε après 200 épisodes : {epsilons[199]:.3f}')
```

---

### Cellule 6 — Boucle d'entraînement

> ⏱️ **Note** : l'entraînement complet (300+ épisodes) prend plusieurs heures. Commencez par 50 épisodes pour vérifier que tout fonctionne, puis augmentez si le temps le permet.

```python
N_EPISODES = 50      # ← augmenter à 300+ pour un vrai apprentissage
MAX_STEPS  = 300     # Steps maximum par épisode
LOG_EVERY  = 10      # Afficher les stats tous les N épisodes

agent = DQNAgent(n_actions=5)
episode_rewards = []
episode_losses  = []

for episode in range(1, N_EPISODES + 1):
    obs, _ = env.reset(seed=episode)
    ep_reward, ep_losses = 0., []

    for step in range(MAX_STEPS):
        # 1. Sélectionner une action (ε-greedy)
        action = agent.select_action(obs)

        # 2. Exécuter l'action dans l'environnement
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 3. Stocker la transition
        agent.store(obs, action, reward, next_obs, done)

        # 4. Entraîner le réseau
        loss = agent.train_step()
        if loss > 0:
            ep_losses.append(loss)

        ep_reward += reward
        obs = next_obs
        if done:
            break

    # Mises à jour post-épisode
    agent.update_epsilon()
    agent.update_target(episode)

    episode_rewards.append(ep_reward)
    episode_losses.append(np.mean(ep_losses) if ep_losses else 0.)

    if episode % LOG_EVERY == 0:
        avg_r = np.mean(episode_rewards[-LOG_EVERY:])
        avg_l = np.mean(episode_losses[-LOG_EVERY:])
        print(f'Ep {episode:4d}/{N_EPISODES} | '
              f'Reward={avg_r:7.1f} | '
              f'Loss={avg_l:.5f} | '
              f'ε={agent.eps:.3f} | '
              f'Buffer={len(agent.buffer):5d}')

print('\n✅ Entraînement terminé')
```

---

### Cellule 7 — Analyse des courbes

```python
# Récompense avec moyenne mobile
plot_reward_curve(episode_rewards, window=10,
                  title='DQN CarRacing-v2 — Récompense par épisode')

# Loss par épisode
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(episode_losses, alpha=0.6, color='coral')
ax.set_xlabel('Épisode')
ax.set_ylabel('Huber Loss')
ax.set_title('Loss DQN par épisode')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Utilisez ce tableau pour diagnostiquer le comportement de votre agent :

| Observation | Diagnostic | Action |
|-------------|-----------|--------|
| Reward stagne autour de −50 | Agent freine trop ou sort de route | Vérifier les actions · augmenter γ |
| Reward très variable, pas de tendance | Pas encore assez d'épisodes | Continuer l'entraînement |
| Loss explose (NaN ou >100) | Gradient exploding | Vérifier le gradient clipping |
| Reward augmente progressivement | ✅ L'agent apprend | Continuer et sauvegarder |
| Reward plateau puis chute | ε trop bas trop vite | Augmenter `eps_decay` |

---

### Cellule 8 — Impact de γ (facteur d'actualisation)

```python
# Visualiser l'impact de gamma sur la valorisation des récompenses futures
gammas = [0.50, 0.90, 0.95, 0.99]
steps  = np.arange(0, 50)

fig, ax = plt.subplots(figsize=(10, 4))
for gamma in gammas:
    discount = gamma ** steps
    ax.plot(steps, discount, label=f'γ = {gamma}', lw=2)

ax.set_xlabel('Horizon temporel (steps dans le futur)')
ax.set_ylabel('Poids de la récompense γᵗ')
ax.set_title('Impact de γ sur la valorisation des récompenses futures')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# TODO : Entraîner le même agent avec γ ∈ {0.90, 0.95, 0.99}
# Pour chaque valeur, tracer la courbe de récompense sur le même graphe
# Conclure : quel γ donne le meilleur résultat sur CarRacing ? Pourquoi ?
```

> 💬 **Question** : `γ = 0.50` signifie qu'une récompense dans 10 steps vaut 0.50¹⁰ ≈ 0.1% de sa valeur immédiate. Qu'est-ce que cela implique pour le comportement de l'agent sur un circuit avec de longs virages ?

---

### Cellule 9 — Sauvegarde et test de l'agent

```python
# Sauvegarder le meilleur modèle
torch.save(agent.policy_net.state_dict(), '/tmp/dqn_carracing_final.pt')
print('Modèle sauvegardé → /tmp/dqn_carracing_final.pt')

# TODO : Charger le modèle et le faire jouer 3 épisodes complets
# Afficher la récompense totale obtenue à chaque épisode
# Comparer avec les récompenses obtenues en début d'entraînement

def play_episode(agent, env, greedy=True):
    """Joue un épisode complet. Si greedy=True, désactive l'exploration."""
    obs, _ = env.reset()
    total_reward = 0.
    steps = 0
    while True:
        if greedy:
            # Exploitation pure : argmax Q(s, a)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = int(agent.policy_net(obs_t).argmax(1).item())
        else:
            action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or steps >= 500:
            break
    return total_reward

print('\nTest de l\'agent (mode greedy) :')
for i in range(3):
    r = play_episode(agent, env, greedy=True)
    print(f'  Épisode {i+1} : reward = {r:.1f}')
```

---

## 4.6 Points de vigilance

### Le buffer doit être suffisamment rempli avant d'entraîner

Entraîner avec trop peu de transitions dans le buffer donne des gradients très bruités. Attendez que le buffer contienne au moins `batch_size * 10` transitions avant de commencer les mises à jour.

```python
# Dans train_step() :
if not self.buffer.is_ready(self.batch_size):
    return 0.0   # Pas encore assez de données
```

### Gradient clipping — indispensable en RL

Les valeurs Q peuvent varier fortement en début d'entraînement. Sans clipping, les gradients explosent :

```python
nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
# Entre loss.backward() et optimizer.step()
```

### `torch.no_grad()` pour le target network

La cible Bellman est calculée avec le `target_net` qui ne doit **jamais** accumuler de gradients :

```python
with torch.no_grad():
    q_next   = self.target_net(next_states).max(dim=1)[0]
    q_target = rewards + self.gamma * q_next * (1 - dones)
```

### La récompense négative initiale est normale

CarRacing-v2 pénalise −0.1 par frame. Un agent qui ne fait rien accumule rapidement une récompense très négative. Les 10 à 50 premiers épisodes sont souvent catastrophiques — c'est attendu.

---

## 4.7 Questions de réflexion

À rédiger dans des **cellules Markdown** du notebook :

1. Pourquoi utilise-t-on deux réseaux (policy et target) plutôt qu'un seul ? Que se passerait-il si on utilisait le même réseau pour calculer la prédiction ET la cible ?

2. Que se passe-t-il si ε reste à 1.0 pendant tout l'entraînement ? Et s'il tombe à 0.0 dès le premier épisode ? Quel est le compromis à trouver ?

3. Comparez la courbe de récompense de votre DQN avec les métriques de vos modèles supervisés (Phase 2 et 3). L'approche RL converge-t-elle plus vite ou plus lentement ? Pourquoi ?

4. *(Optionnel)* Testez `gamma` ∈ {0.90, 0.95, 0.99}. Lequel donne les meilleurs résultats sur CarRacing ? Expliquez le lien entre γ et la longueur des virages du circuit.

---

## 4.8 Bilan du projet fil rouge

> ✅ **Ce que vous avez construit en 14 heures**
>
> Un pipeline IA complet de la perception à la décision autonome :
>
> ```
>                    Frame RGB
>                   (96×96×3)
>                       │
>              ┌─────────▼─────────┐
>              │  Phase 1 — CNN    │  Perception visuelle
>              │  PerceptionCNN    │  Classifier la route
>              └─────────┬─────────┘
>                        │  Features extraites
>           ┌────────────┼────────────┐
>           ▼            ▼            ▼
>    ┌────────────┐ ┌─────────┐ ┌──────────┐
>    │ Phase 2    │ │ Phase 3 │ │ Phase 4  │
>    │ DrivingMLP │ │  LSTM   │ │   DQN    │
>    │ Supervisé  │ │Séquentiel│ │   RL    │
>    └────────────┘ └─────────┘ └──────────┘
>    Imitation      Mémoire      Autonomie
> ```
>
> Chaque phase a introduit un concept fondamental du deep learning en industrie, ancré dans un même environnement visuel et interactif. Le CNN de la Phase 1 était l'encodeur de toutes les phases suivantes — c'est exactement l'architecture utilisée dans les véhicules autonomes réels.

### Pour aller plus loin (pistes MSPR)

- **Double DQN** : utiliser le policy_net pour sélectionner l'action et le target_net pour l'évaluer → réduit la surestimation des Q-valeurs
- **Dueling DQN** : séparer la valeur d'état V(s) et l'avantage A(s,a) → convergence plus stable
- **Prioritized Experience Replay** : échantillonner les transitions avec la plus grande erreur TD en priorité → déjà disponible dans `utils/replay_buffer.py`
- **Frame stacking** : empiler 4 frames consécutives → donner une notion de vitesse et de direction à l'agent sans LSTM

---

## Ressources utiles

- [DQN paper original — Mnih et al. 2015](https://www.nature.com/articles/nature14236) ⭐
- [Spinning Up in Deep RL — OpenAI](https://spinningup.openai.com)
- [PyTorch nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Gymnasium CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/)
- Sutton & Barto — *Reinforcement Learning: An Introduction* · MIT Press 2018 (libre de droit)

---

*NRLE822 · Projet fil rouge · Phase 4/4 · EPSI 2025-2026*
