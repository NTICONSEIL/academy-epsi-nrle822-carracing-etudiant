# Prise en main de l'environnement CarRacing-v3
## NRLE822 · Projet fil rouge · Séance d'introduction

> **Module** NRLE822 · S8 &nbsp;|&nbsp; **Durée** 1h &nbsp;|&nbsp; **Niveau** Débutant  
> **Notebook** `00_prise_en_main.ipynb` &nbsp;|&nbsp; **Prérequis** Aucun  
> *Mastère SIN — Expert en Intelligence Artificielle · EPSI 2025-2026*

---

## Objectifs

À la fin de cette séance, vous serez capables de :

- installer et vérifier l'environnement de travail (Gymnasium, PyTorch) ;
- comprendre le cycle fondamental `observation → action → récompense → nouvel état` ;
- inspecter les espaces d'observation et d'action de CarRacing-v3 ;
- visualiser ce que « voit » la voiture à chaque pas de temps ;
- exécuter un agent aléatoire et mesurer son score ;
- produire une vidéo MP4 d'un épisode sur Google Colab.

> Cette séance ne nécessite aucune connaissance préalable en deep learning. Elle pose les fondations communes aux 4 phases du projet fil rouge.

---

## 1. Installation des dépendances

Cette cellule installe toutes les bibliothèques nécessaires au projet. Elle est conçue pour Google Colab et doit être exécutée en premier.

```python
# ── Outils système nécessaires à Box2D et à la génération de vidéos ──────────
!apt-get update -qq
!apt-get install -y -qq swig ffmpeg xvfb

# ── Bibliothèques Python ──────────────────────────────────────────────────────
!pip install -q "gymnasium[box2d]" moviepy
!pip install -q torch==2.2.0 torchvision==0.17.0
!pip install -q numpy pandas matplotlib seaborn tqdm pillow
```

> ⚠️ **Erreur `Box2D not found` ?** Assurez-vous que `swig` est bien installé avant `gymnasium[box2d]`. L'ordre des commandes ci-dessus est intentionnel.

---

## 2. Imports et configuration

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from IPython.display import HTML, display
from base64 import b64encode
from gymnasium.wrappers import RecordVideo

# ── Reproductibilité ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
print(f'Gymnasium : {gym.__version__}')
print(f'PyTorch   : {torch.__version__}')
```

---

## 3. Création de l'environnement et exploration des espaces

CarRacing-v3 expose deux objets importants que vous retrouverez dans toutes les phases du projet.

```python
# render_mode="rgb_array" : l'environnement retourne les frames en mémoire
# Ne jamais utiliser render_mode="human" sur Colab ni pendant l'entraînement
env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)

obs, info = env.reset(seed=SEED)

print('─── Espace d\'observation ───────────────────────────────')
print(f'  Type  : {env.observation_space}')
print(f'  Shape : {obs.shape}    ← (hauteur, largeur, canaux RGB)')
print(f'  dtype : {obs.dtype}')
print(f'  Min   : {obs.min()}  |  Max : {obs.max()}')

print('\n─── Espace d\'action ─────────────────────────────────────')
print(f'  Type  : {env.action_space}')
print(f'  Shape : {env.action_space.shape}  ← [steering, gas, brake]')
print(f'  Low   : {env.action_space.low}')
print(f'  High  : {env.action_space.high}')

env.close()
```

> 💬 **Question** : l'espace d'action est dit « continu » (`Box`). Qu'est-ce que cela implique par rapport à un espace discret (`Discrete`) comme dans CartPole ?

---

## 4. Visualiser ce que voit la voiture

L'agent ne reçoit ni la vitesse, ni la position GPS, ni l'angle du volant. Il ne perçoit que cette image de 96 × 96 pixels — exactement comme un conducteur humain dans un jeu vidéo.

```python
env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
obs, info = env.reset(seed=SEED)

# Avancer de quelques steps pour sortir de l'animation de démarrage
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.5, 0.0], dtype=np.float32)  # tout droit, accélère légèrement
    )

# Grille 2×4 : 8 frames à des instants différents
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle('Ce que voit la voiture — 8 frames consécutives', fontsize=13)

for ax in axes.flatten():
    ax.imshow(obs)
    ax.axis('off')
    # Avancer de 10 steps entre chaque frame
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        if terminated or truncated:
            obs, _ = env.reset(seed=SEED)
            break

plt.tight_layout()
plt.show()

env.close()
```

> 💬 **Question** : quelles informations visuelles sont disponibles dans ces frames ? Identifiez au moins 4 éléments (couleur de la route, herbe, capot, indicateurs...).

---

## 5. Le cycle fondamental : observation → action → récompense

Voici la boucle de base que vous retrouverez dans toutes les phases du projet. Chaque itération est appelée un **pas de temps** (step).

```python
env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
obs, info = env.reset(seed=SEED)

print(f'{"Step":>4}  {"Action":^32}  {"Reward":>8}  {"Cum. reward":>12}')
print('─' * 65)

cumulative_reward = 0.0

for step in range(20):
    # Action : [steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

    print(f'{step:>4}  [{action[0]:+.3f}, {action[1]:.3f}, {action[2]:.3f}]'
          f'  {reward:>8.4f}  {cumulative_reward:>12.4f}')

    if terminated or truncated:
        print('Episode terminé.')
        break

env.close()
```

> 💬 **Question** : la récompense est souvent légèrement négative (−0.1) même quand la voiture avance. Consultez la documentation de CarRacing-v2 et expliquez pourquoi.

---

## 6. Actions prédéfinies

Avant d'utiliser un modèle appris, il est utile de comprendre l'effet de chaque action.

```python
actions_predefinies = {
    'Tout droit — accélère' : np.array([ 0.0, 1.0, 0.0], dtype=np.float32),
    'Virage gauche'         : np.array([-1.0, 0.3, 0.0], dtype=np.float32),
    'Virage droite'         : np.array([ 1.0, 0.3, 0.0], dtype=np.float32),
    'Freinage'              : np.array([ 0.0, 0.0, 1.0], dtype=np.float32),
    'Inaction'              : np.array([ 0.0, 0.0, 0.0], dtype=np.float32),
}

env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
obs, _ = env.reset(seed=SEED)

# Avancer légèrement pour voir la route
for _ in range(40):
    env.step(np.array([0.0, 0.8, 0.0], dtype=np.float32))

fig, axes = plt.subplots(1, len(actions_predefinies), figsize=(16, 3))
fig.suptitle('Effet visuel de 5 actions après 5 steps', fontsize=12)

for ax, (label, action) in zip(axes, actions_predefinies.items()):
    # Sauvegarder l'état courant, appliquer 5 steps avec cette action
    env_copy_obs, _ = env.reset(seed=SEED)
    for _ in range(40):
        env_copy_obs, _, _, _, _ = env.step(np.array([0.0, 0.8, 0.0], dtype=np.float32))
    for _ in range(5):
        env_copy_obs, _, _, _, _ = env.step(action)
    ax.imshow(env_copy_obs)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()
env.close()
```

---

## 7. Fonction utilitaire — afficher une vidéo dans Colab

```python
def show_video(video_folder: str = 'videos', width: int = 500) -> None:
    """Affiche la dernière vidéo MP4 trouvée dans video_folder."""
    import glob
    mp4_files = sorted(glob.glob(f'{video_folder}/**/*.mp4', recursive=True))
    if not mp4_files:
        print('Aucune vidéo trouvée dans', video_folder)
        return
    latest = mp4_files[-1]
    print(f'Vidéo : {latest}')
    data_url = 'data:video/mp4;base64,' + b64encode(open(latest, 'rb').read()).decode()
    display(HTML(f'<video width="{width}" controls><source src="{data_url}" type="video/mp4"></video>'))
```

---

## 8. Agent aléatoire — baseline minimale

Un agent qui choisit ses actions au hasard sert de **baseline** : tout modèle appris doit faire mieux.

```python
os.makedirs('videos/random_agent', exist_ok=True)

env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
env = RecordVideo(
    env,
    video_folder='videos/random_agent',
    episode_trigger=lambda episode_id: True,
    name_prefix='random'
)

obs, _ = env.reset(seed=SEED)
total_reward = 0.0
rewards_par_step = []

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    rewards_par_step.append(reward)

    if terminated or truncated:
        break

env.close()
print(f'Score agent aléatoire : {total_reward:.2f}  ({len(rewards_par_step)} steps)')

show_video('videos/random_agent')
```

---

## 9. Courbe des récompenses

Visualiser la courbe de récompense cumulée permet de comprendre quand la voiture « sort de la route » (chutes de récompense).

```python
cumulative = np.cumsum(rewards_par_step)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(rewards_par_step, lw=0.8, color='steelblue', alpha=0.7)
axes[0].axhline(0, color='gray', lw=0.5, ls='--')
axes[0].set(xlabel='Step', ylabel='Récompense', title='Récompense par step')
axes[0].grid(alpha=0.3)

axes[1].plot(cumulative, lw=1.5, color='darkorange')
axes[1].set(xlabel='Step', ylabel='Récompense cumulée', title='Récompense cumulée')
axes[1].grid(alpha=0.3)

plt.suptitle(f'Agent aléatoire — Score final : {total_reward:.2f}', fontsize=12)
plt.tight_layout()
plt.show()
```

> 💬 **Question** : à quels moments la récompense par step chute-t-elle fortement ? Que se passe-t-il dans l'environnement à ces instants ?

---

## 10. Baseline statistique sur plusieurs épisodes

Un seul épisode ne suffit pas : les circuits de CarRacing-v2 sont générés aléatoirement à chaque `reset()`. Il faut plusieurs épisodes pour estimer le score moyen d'une politique.

```python
def run_episode(seed: int = None, max_steps: int = 1000) -> float:
    """Exécute un épisode avec un agent aléatoire et retourne le score total."""
    env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(max_steps):
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total += r
        if terminated or truncated:
            break
    env.close()
    return total


N_EPISODES = 5
scores = [run_episode(seed=SEED + i) for i in range(N_EPISODES)]

print('Scores par épisode :', [f'{s:.1f}' for s in scores])
print(f'Moyenne            : {np.mean(scores):.2f}')
print(f'Écart-type         : {np.std(scores):.2f}')
print(f'Min / Max          : {np.min(scores):.2f} / {np.max(scores):.2f}')

plt.figure(figsize=(7, 4))
plt.bar(range(N_EPISODES), scores, color='steelblue', alpha=0.8)
plt.axhline(np.mean(scores), color='red', ls='--', lw=1.5, label=f'Moyenne = {np.mean(scores):.1f}')
plt.xlabel('Épisode')
plt.ylabel('Score total')
plt.title('Scores de l\'agent aléatoire sur 5 épisodes')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

> 💬 **Question** : pourquoi l'écart-type est-il élevé ? La génération aléatoire des circuits est-elle une bonne ou une mauvaise chose pour l'évaluation d'un agent ?

---

## 11. Questions de synthèse

Rédigez vos réponses dans des **cellules Markdown** après chaque section correspondante. Ces réponses ne sont pas évaluées dans cette séance, mais elles préparent les questions des Phases 1 à 4.

1. Décrivez le cycle `observation → action → récompense → nouvel état` dans vos propres mots, en utilisant un exemple concret tiré de CarRacing-v2.

2. L'espace d'observation est une image 96 × 96 × 3. Calculez le nombre de valeurs distinctes dans une seule frame. Pourquoi est-il impossible d'utiliser une table de correspondance (lookup table) pour mémoriser la politique optimale ?

3. Comparez le score moyen de l'agent aléatoire à votre intuition : est-il meilleur ou moins bon que ce que vous attendiez ? Expliquez.

4. Identifiez dans le code la différence entre `terminated` et `truncated`. Dans quel cas chacun devient-il `True` dans CarRacing-v2 ?

5. *(Réflexion)* Un agent humain obtient en général un score entre 700 et 900. Quelles informations supplémentaires utilise-t-il que notre agent aléatoire n'exploite pas ?

---

## Transition vers la Phase 1

> À ce stade, vous avez compris ce que voit et fait la voiture à chaque instant. Vous avez également une **baseline** : le score d'un agent aléatoire, que tout modèle appris devra dépasser.
>
> La **Phase 1** va exploiter ces frames brutes pour entraîner un premier réseau de neurones convolutionnel (CNN) capable de reconnaître le type de route visible : ligne droite, virage à gauche ou virage à droite.
>
> Ce CNN deviendra l'**encodeur visuel partagé** par toutes les phases suivantes.

---

*NRLE822 · Projet fil rouge · Séance d'introduction · EPSI 2025-2026*
