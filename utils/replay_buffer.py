"""
utils/replay_buffer.py
----------------------
Experience Replay Buffer pour l'algorithme DQN.

Le replay buffer stocke les transitions (s, a, r, s', done) et
permet de les ré-échantillonner aléatoirement pour briser les
corrélations temporelles lors de l'entraînement.

Référence : Mnih et al. (2015) "Human-level control through deep RL"
"""

import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Transition:
    """Une transition (s, a, r, s', done) dans l'environnement."""
    state: np.ndarray        # Observation à l'instant t
    action: int              # Action discrète choisie
    reward: float            # Récompense reçue
    next_state: np.ndarray   # Observation à l'instant t+1
    done: bool               # True si épisode terminé


class ReplayBuffer:
    """
    Buffer circulaire pour l'experience replay.
    
    Stocke les N dernières transitions et permet de tirer
    un mini-batch aléatoire pour l'entraînement du réseau Q.
    
    Paramètres
    ----------
    capacity : int
        Nombre maximum de transitions stockées.
        Quand le buffer est plein, les plus anciennes sont écrasées.
    
    Exemple
    -------
    >>> buffer = ReplayBuffer(capacity=10_000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> len(buffer)
    1
    >>> states, actions, rewards, next_states, dones = buffer.sample(32)
    """

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Ajoute une transition au buffer."""
        self.buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Tire aléatoirement batch_size transitions du buffer.
        
        Retourne
        --------
        states      : (batch_size, *obs_shape)  float32
        actions     : (batch_size,)              int64
        rewards     : (batch_size,)              float32
        next_states : (batch_size, *obs_shape)  float32
        dones       : (batch_size,)              float32 (0.0 ou 1.0)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Pas assez de transitions : {len(self.buffer)} < {batch_size}. "
                "Continuez à collecter des expériences."
            )
        
        transitions = random.sample(self.buffer, batch_size)
        
        states      = np.stack([t.state      for t in transitions])
        actions     = np.array([t.action     for t in transitions], dtype=np.int64)
        rewards     = np.array([t.reward     for t in transitions], dtype=np.float32)
        next_states = np.stack([t.next_state for t in transitions])
        dones       = np.array([t.done       for t in transitions], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Retourne True si le buffer contient assez de transitions pour un batch."""
        return len(self.buffer) >= batch_size

    @property
    def fill_ratio(self) -> float:
        """Proportion du buffer rempli (0.0 → 1.0)."""
        return len(self.buffer) / self.capacity


class PrioritizedReplayBuffer:
    """
    Replay buffer avec priorités (Prioritized Experience Replay).
    
    Les transitions avec une grande erreur TD sont échantillonnées
    plus souvent. Version simplifiée pour usage pédagogique.
    
    Référence : Schaul et al. (2015) "Prioritized Experience Replay"
    
    Paramètres
    ----------
    capacity : int   - Capacité maximale
    alpha    : float - Exposant de priorité (0 = uniforme, 1 = full priority)
    beta     : float - Exposant de correction IS (importance sampling)
    
    Note formateur : Cette classe est présentée comme extension avancée
    dans le notebook 04. Les apprenants n'ont pas à l'implémenter.
    """

    def __init__(self, capacity: int = 10_000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        transition = Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Poids d'importance sampling
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        transitions = [self.buffer[i] for i in indices]
        states      = np.stack([t.state      for t in transitions])
        actions     = np.array([t.action     for t in transitions], dtype=np.int64)
        rewards     = np.array([t.reward     for t in transitions], dtype=np.float32)
        next_states = np.stack([t.next_state for t in transitions])
        dones       = np.array([t.done       for t in transitions], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """Met à jour les priorités après calcul des erreurs TD."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6  # Epsilon pour éviter priorité nulle

    def __len__(self) -> int:
        return self.size
