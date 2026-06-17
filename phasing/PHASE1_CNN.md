# Conduite Autonome avec des Réseaux de Neurones
## NRLE822 · Projet fil rouge · CarRacing-v2

> **Module** NRLE822 · S8 &nbsp;|&nbsp; **Durée** 14h (4 phases) &nbsp;|&nbsp; **Niveau** Intermédiaire &nbsp;|&nbsp; **Livrable** Notebook Jupyter  
> *Mastère SIN — Expert en Intelligence Artificielle · EPSI 2025-2026*

---

# Partie 1 — Présentation du projet

## 1.1 Contexte

Vous prenez le rôle d'ingénieur·e IA au sein d'une startup de mobilité autonome. Votre mission : concevoir un pipeline complet permettant à une voiture virtuelle d'apprendre à conduire, en utilisant les techniques de deep learning les plus utilisées en industrie aujourd'hui.

L'environnement de simulation choisi est **CarRacing-v2** (Gymnasium / Farama Foundation), un standard open-source utilisé dans la recherche en IA. À chaque pas de temps, la voiture perçoit une image RGB 96×96 pixels et doit décider d'une action [direction, accélération, freinage].

> *Ce projet n'est pas un ensemble d'exercices indépendants : les 4 phases s'enchaînent et se réutilisent. Le CNN que vous construisez en Phase 1 devient l'encodeur visuel des Phases 2, 3 et 4.*

---

## 1.2 Pourquoi ce projet ?

Les techniques apprises ici correspondent directement aux métiers de l'IA en 2025 :

| Phase | Applications industrielles |
|-------|---------------------------|
| **Phase 1 — CNN** | Computer Vision · Détection d'objets (YOLO) · Imagerie médicale · Contrôle qualité |
| **Phase 2 — Supervisé** | Trading algorithmique · Maintenance prédictive · Scoring de crédit · Météorologie |
| **Phase 3 — LSTM** | Traduction automatique · Prévision d'énergie · Détection d'anomalies · Finance |
| **Phase 4 — DQN / RL** | Robotique industrielle · AlphaGo · Gestion de datacenters · Logistique |

---

## 1.3 L'environnement CarRacing-v2

À chaque pas de temps, l'environnement fournit :

- **Observation** : image RGB 96×96 pixels — ce que « voit » la voiture
- **Action continue** : `[steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]`
- **Récompense** : +score par portion de route visitée · −0.1 par frame

> ⚠️ **Important** — N'utilisez jamais `render_mode="human"` lors de l'entraînement : cela ouvre une fenêtre graphique et divise la vitesse par 10. Utilisez toujours `render_mode=None` dans vos notebooks.

---

## 1.4 Structure des 4 phases

| Phase | Contenu | Durée | Compétences |
|-------|---------|-------|-------------|
| **Phase 1 — Perception (CNN)** | Classifier le type de route visible (droite / virage G / virage D) | 3h | CDPEIA 2.3 · 2.5 |
| **Phase 2 — Imitation (Supervisé)** | Prédire les actions d'un pilote à partir des observations | 4h | CDPEIA 2.5 · 2.6 · 2.7 |
| **Phase 3 — Mémoire (LSTM)** | Anticiper les virages grâce à l'historique des frames | 4h | CDPEIA 2.5 · 2.6 |
| **Phase 4 — Agent (DQN)** | Apprendre à conduire par essai-erreur dans l'environnement | 3h | CDPEIA 2.5 · 2.6 · 2.7 |

---

## 1.5 Livrable et évaluation (MSPR TPRE822)

Vous rendez un **unique Notebook Jupyter** exécutable de bout en bout, documenté et commenté, présentant votre solution complète pour les 4 phases.

| Critère | Indicateur | Points |
|---------|-----------|--------|
| Architecture et conception du modèle | Choix justifié, pipeline documenté, flux de données clair | **6 pts** |
| Procédure d'entraînement et split | Split temporel justifié, hyperparamètres explorés | **6 pts** |
| Analyse des performances | Métriques pertinentes, courbes commentées, comparaison de modèles | **5 pts** |
| Qualité du notebook | Lisible, reproductible, seeds fixés, conclusion rédigée | **3 pts** |

**Checklist minimale de rendu :**

- [ ] Kernel Restart → Run All s'exécute sans erreur
- [ ] Seeds fixés (`SEED = 42`) en tête de chaque notebook
- [ ] Split train/val/test documenté et justifié dans une cellule Markdown
- [ ] Au moins une courbe de loss commentée par phase
- [ ] Au moins deux modèles ou configurations comparés
- [ ] Conclusion rédigée : forces, limites, améliorations envisagées

---

## 1.6 Installation et démarrage

### Cloner le repo

```bash
git clone https://github.com/NTICONSEIL/academy-epsi-nrle822-carracing-etudiant.git
cd academy-epsi-nrle822-carracing-etudiant
```

### Créer l'environnement

```bash
conda create -n carracing python=3.10 -y
conda activate carracing

# Linux uniquement
sudo apt-get install -y swig libgl1-mesa-glx

pip install swig
pip install gymnasium[box2d]==0.29.1
pip install torch==2.2.0 torchvision==0.17.0
pip install numpy pandas matplotlib seaborn jupyter ipywidgets tqdm pillow pygame
```

### Vérifier l'installation

```bash
python utils/check_install.py

# Résultat attendu :
# ✅ numpy OK     ✅ torch OK     ✅ gymnasium OK
# ✅ CarRacing-v2 créé et réinitialisé avec succès
```

> 🔴 **Erreur `Box2D not found` ?**  
> ```bash
> pip install swig
> pip install gymnasium[box2d]
> # Si ça persiste sur Linux : sudo apt-get install swig
> ```
> Consultez la section Erreurs fréquentes du `README.md` si le problème persiste.

---

# Partie 2 — Phase 1 : Perception de l'environnement (CNN)

> **Objectif** : Construire un réseau de neurones convolutionnel (CNN) capable de classifier le type de route visible dans une frame de CarRacing-v2 : **ligne droite, virage à gauche, virage à droite**. À la fin de cette phase, vous serez en mesure de visualiser ce que « voit » votre réseau grâce aux feature maps.
>
> Durée estimée : **3h** · Notebook : `notebooks/01_CNN_perception.ipynb`

---

## 2.1 Rappels théoriques

### Le neurone artificiel

Un neurone calcule une somme pondérée de ses entrées, ajoute un biais, puis applique une **fonction d'activation** pour introduire la non-linéarité :

```
sortie = activation( Σ (wᵢ · xᵢ) + b )
```

Sans fonction d'activation, un réseau profond serait équivalent à une seule transformation linéaire — inutile.

| Fonction | Formule | Cas d'usage |
|----------|---------|-------------|
| **ReLU** | `max(0, x)` | Rapide, pas de saturation pour x>0 · Standard dans les CNNs |
| **Sigmoid** | `1/(1+e⁻ˣ)` | Sortie ∈ [0,1] · Classification binaire (dernière couche) |
| **Tanh** | `tanh(x)` | Sortie ∈ [−1,1], centrée en 0 · Moins utilisée depuis ReLU |
| **Leaky ReLU** | `x si x>0, 0.01x sinon` | Évite les neurones morts · Variante de ReLU |

### Pourquoi un CNN plutôt qu'un MLP ?

Un MLP appliqué à une image 96×96×3 aplatirait l'entrée en **27 648 valeurs**, perdant toute information spatiale. La première couche cachée (128 neurones) aurait **27 648 × 128 = 3 538 944 paramètres** — trop lourd et inefficace.

Un CNN applique des **filtres glissants (convolutions)** qui détectent des patterns locaux (bords, textures, formes) puis les combinent hiérarchiquement. Avantages : partage des poids, invariance à la translation, beaucoup moins de paramètres.

### Les opérations clés d'un CNN

- **Couche de convolution** : filtre w×h glissant sur l'image, produit une feature map
- **BatchNormalization** : normalise les activations → entraînement plus stable et plus rapide
- **ReLU** : activation non-linéaire appliquée après chaque convolution
- **Dropout** : désactive aléatoirement p% des neurones → régularisation, réduit l'overfitting
- **Couche fully-connected (FC)** : classifie les features extraites

---

## 2.2 Architecture à implémenter

Vous allez implémenter la classe `PerceptionCNN` dans le notebook. Voici l'architecture cible :

| Couche | Détails | Sortie |
|--------|---------|--------|
| Entrée | Image RGB normalisée [0,1] | `(batch, 3, 96, 96)` |
| Conv1 + BN + ReLU | 8 filtres 8×8, stride=4 | `(batch, 8, 23, 23)` |
| Conv2 + BN + ReLU | 16 filtres 4×4, stride=2 | `(batch, 16, 10, 10)` |
| Conv3 + BN + ReLU | 32 filtres 3×3, stride=1 | `(batch, 32, 8, 8)` |
| Flatten | — | `(batch, 2048)` |
| Dropout(p=0.3) + FC(128) + ReLU | — | `(batch, 128)` |
| FC(n_classes) | Logits pour 3 classes | `(batch, 3)` |

**Formule de calcul des dimensions :**

```
sortie = floor( (entrée − kernel_size) / stride ) + 1

Conv1 : floor( (96 − 8) / 4 ) + 1 = 23  ✓
Conv2 : floor( (23 − 4) / 2 ) + 1 = 10  ✓
Conv3 : floor( (10 − 3) / 1 ) + 1 = 8   ✓
```

---

## 2.3 Travail à réaliser (notebook 01)

### Cellule 1 — Configuration

Importez les bibliothèques et fixez les seeds pour la reproductibilité :

```python
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym

# TODO : Fixer les seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)  # Indispensable pour la reproductibilité

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
```

---

### Cellule 2 — Exploration de l'environnement

Créez l'environnement et visualisez au moins 8 frames variées.

```python
env = gym.make('CarRacing-v2', render_mode=None, continuous=True)

# Afficher les espaces d'observation et d'action
print('Observation :', env.observation_space)   # Box(96, 96, 3)
print('Action      :', env.action_space)        # Box(3,)

# TODO : Visualiser 8 frames (grid 2×4)
# Astuce : env.step(env.action_space.sample()) pour avancer
```

> 💬 **Question** : quelles informations visuelles sont disponibles dans ces frames ? (couleur de la route, herbe, capot…)

---

### Cellule 3 — Fonctions d'activation

Tracez les 4 fonctions d'activation sur `x ∈ [−4, 4]` dans une grille 1×4.

```python
x = np.linspace(-4, 4, 200)

# TODO : Définir et tracer ReLU, Sigmoid, Tanh, Leaky ReLU
# Attendu : 4 graphes côte à côte avec titre et grille
```

> 💬 **Question** : pourquoi ReLU est-il préféré à Sigmoid dans les couches cachées d'un CNN ?

---

### Cellule 4 — Implémentation de `PerceptionCNN`

Implémentez la classe `PerceptionCNN`. Vérifiez ensuite les dimensions avec un batch factice :

```python
class PerceptionCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # TODO : Définir les couches
        # self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)
        # self.bn1   = nn.BatchNorm2d(8)
        # ...

    def forward(self, x):
        # TODO : Implémenter le forward pass
        pass

# Vérification obligatoire — ne modifiez pas cette cellule
model = PerceptionCNN(n_classes=3)
dummy = torch.zeros(4, 3, 96, 96)
with torch.no_grad():
    out = model(dummy)
print(f'Entrée {dummy.shape} → Sortie {out.shape}')  # Attendu: torch.Size([4, 3])
assert out.shape == (4, 3), 'Dimensions incorrectes !'
```

---

### Cellule 5 — Collecte des données et DataLoaders

Collectez environ 1 200 frames avec un agent cyclique (20 steps gauche / 20 steps droite / 20 steps tout droit). Créez un Dataset PyTorch et les DataLoaders avec un split **70% train / 15% val / 15% test**.

> 💡 **Conversion des dimensions pour PyTorch**  
> Gymnasium retourne des images au format `(H, W, C)`. PyTorch attend `(C, H, W)`.  
> Utilisez `frames.transpose(0, 3, 1, 2)` lors de la création du Dataset.  
> Normalisez les pixels vers [0, 1] en divisant par 255.

```python
class CarFrameDataset(Dataset):
    def __init__(self, frames, labels):
        # TODO : Convertir (N,H,W,C) → (N,C,H,W) et normaliser
        pass
    def __len__(self): ...
    def __getitem__(self, i): ...

# TODO : Collecter les frames avec l'agent cyclique
# TODO : Split 70/15/15 et créer les DataLoaders (batch_size=32)
```

---

### Cellule 6 — Boucle d'entraînement

Implémentez la boucle d'entraînement. Utilisez `CrossEntropyLoss` et `Adam(lr=1e-3)`. Sauvegardez le meilleur modèle (val_loss la plus basse).

```python
model     = PerceptionCNN(n_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()       # Perte pour classification multi-classe
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# TODO : Boucle for epoch in range(N_EPOCHS):
#   - Phase train : forward → loss → backward → step
#   - Phase val   : forward uniquement (torch.no_grad())
#   - Stocker train_loss, val_loss, train_acc, val_acc
#   - Sauvegarder si val_loss < best_val_loss
```

**Les 4 étapes du backward pass :**

```python
optimizer.zero_grad()    # 1. Réinitialiser les gradients accumulés
loss = criterion(pred, target)  # 2. Calculer la perte
loss.backward()          # 3. Calculer les gradients par rétropropagation
optimizer.step()         # 4. Mettre à jour les poids : w ← w − lr × ∂L/∂w
```

---

### Cellule 7 — Courbes d'apprentissage

Affichez les courbes de loss et d'accuracy (train vs val) sur deux graphes côte à côte. Puis diagnostiquez votre modèle :

| Observation | Diagnostic et action |
|-------------|---------------------|
| `val_loss >> train_loss` | **Surapprentissage** → augmenter Dropout, réduire le modèle, ajouter des données |
| `train_loss` élevée et stagne | **Sous-apprentissage** → réseau plus profond, learning rate plus élevé |
| Train et val se rejoignent bas | **Bonne convergence** ✓ |
| Loss qui oscille fortement | **Learning rate trop élevé** → essayer `1e-4` |

---

### Cellule 8 — Visualisation des feature maps

Utilisez `register_forward_hook` pour capturer les activations de la 1ère couche convolutionnelle et affichez les 8 feature maps côte à côte avec l'image originale.

```python
model.eval()
activations = {}

# Enregistrer un hook sur conv1
hook = model.conv1.register_forward_hook(
    lambda m, i, o: activations.update({'conv1': o.detach().cpu()})
)

with torch.no_grad():
    model(sample_input.to(DEVICE))

hook.remove()  # Toujours supprimer le hook après usage

# TODO : Visualiser activations['conv1'][0]  →  shape (8, 23, 23)
# Utilisez imshow avec cmap='RdBu_r' pour voir les activations positives/négatives
```

---

## 2.4 Questions de réflexion

Ces questions doivent être rédigées dans des **cellules Markdown** de votre notebook, après chaque cellule de code correspondante. Elles font partie du livrable évalué.

1. Pourquoi un CNN est-il plus adapté qu'un MLP pour traiter des images de 96×96 pixels ? Comparez le nombre de paramètres des deux approches.

2. Observez vos feature maps. Que détectent les différents filtres de la Conv1 ? Pouvez-vous associer certains filtres à des concepts visuels (bords, couleurs, textures) ?

3. Vos courbes de loss montrent-elles du surapprentissage ? Si oui, quelle modification architecturale ou d'entraînement proposez-vous ? Testez-la et comparez les résultats.

4. *(Optionnel — fortement recommandé pour la note)* Quel serait l'impact de supprimer BatchNorm ? Supprimez-le sur une copie du modèle, entraînez et comparez les courbes.

---

## 2.5 Transition vers la Phase 2

> ⚠️ **Ce que la Phase 1 ne peut pas faire**
>
> Votre CNN classifie bien le type de route, mais il ne sait pas **quelle action choisir**. Il voit une image et dit « c'est un virage à droite » — mais il ne sait pas à quel point tourner, ni si freiner est utile.
>
> La Phase 2 va résoudre ce problème : on va apprendre au modèle à **prédire directement les valeurs de [steering, gas, brake]** en imitant un pilote, via l'apprentissage supervisé.
>
> 💬 **Question de réflexion** : si le steering est une valeur continue ∈ [−1, 1], quel type de problème résout-on : classification ou régression ? Quelle loss function utiliser ?

---

## 2.6 Ressources utiles

- [Documentation PyTorch nn.Conv2d](https://pytorch.org/docs/stable/nn.html)
- [CS231n — Convolutional Neural Networks](https://cs231n.github.io)
- [Playground TensorFlow — visualisation interactive](https://playground.tensorflow.org)
- Azencott — *Introduction au Machine Learning* (2e éd.) · Dunod 2022 · Bibliothèque ENI

---

*NRLE822 · Projet fil rouge · Niveau intermédiaire · EPSI 2025-2026*
