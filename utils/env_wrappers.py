"""
utils/env_wrappers.py
---------------------
Wrappers Gymnasium pour simplifier l'utilisation de CarRacing-v2
dans les notebooks apprenants.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Convertit les observations RGB (96, 96, 3) en niveaux de gris (96, 96, 1).
    
    Utile pour réduire la dimensionnalité et accélérer l'entraînement
    lorsque la couleur n'est pas informative.
    
    Exemple
    -------
    >>> env = gym.make("CarRacing-v2")
    >>> env = GrayscaleWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs.shape
    (96, 96, 1)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(96, 96, 1),
            dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Coefficients de luminance ITU-R BT.601
        gray = (0.299 * obs[:, :, 0] +
                0.587 * obs[:, :, 1] +
                0.114 * obs[:, :, 2])
        return gray[:, :, np.newaxis].astype(np.uint8)


class ResizeWrapper(gym.ObservationWrapper):
    """
    Redimensionne les observations à une taille cible.
    
    Utile pour réduire la charge computationnelle sur CPU.
    
    Paramètres
    ----------
    size : tuple (H, W) - Taille cible. Par défaut (42, 42).
    
    Exemple
    -------
    >>> env = gym.make("CarRacing-v2")
    >>> env = ResizeWrapper(env, size=(42, 42))
    >>> obs, _ = env.reset()
    >>> obs.shape
    (42, 42, 3)
    """

    def __init__(self, env: gym.Env, size: tuple = (42, 42)):
        super().__init__(env)
        self.size = size
        h, w = size
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(h, w, env.observation_space.shape[2]),
            dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(obs)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


class NormalizeWrapper(gym.ObservationWrapper):
    """
    Normalise les pixels de [0, 255] vers [0.0, 1.0].
    
    À appliquer après GrayscaleWrapper ou ResizeWrapper.
    Retourne des float32.
    
    Exemple
    -------
    >>> env = gym.make("CarRacing-v2")
    >>> env = NormalizeWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs.dtype
    float32
    >>> obs.min(), obs.max()
    (0.0, 1.0)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low.astype(np.float32) / 255.0
        high = env.observation_space.high.astype(np.float32) / 255.0
        self.observation_space = spaces.Box(
            low=low, high=high,
            dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Empile N frames consécutives pour créer un contexte temporel.
    
    Utile pour les architectures CNN sans LSTM : donner une notion
    de mouvement au réseau en lui montrant plusieurs frames à la fois.
    
    Paramètres
    ----------
    n_frames : int - Nombre de frames à empiler. Par défaut 4.
    
    Exemple
    -------
    >>> env = gym.make("CarRacing-v2")
    >>> env = GrayscaleWrapper(env)
    >>> env = FrameStackWrapper(env, n_frames=4)
    >>> obs, _ = env.reset()
    >>> obs.shape
    (96, 96, 4)
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames = None
        
        obs_shape = env.observation_space.shape
        # Empiler sur le dernier axe (canaux)
        new_shape = obs_shape[:-1] + (obs_shape[-1] * n_frames,)
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialiser le buffer avec la première frame répétée
        self._frames = [obs] * self.n_frames
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.pop(0)
        self._frames.append(obs)
        return np.concatenate(self._frames, axis=-1)


class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Discrétise l'espace d'action continu de CarRacing-v2.
    
    Transforme l'espace continu [steering, gas, brake] en
    un ensemble fini d'actions discrètes. Nécessaire pour DQN.
    
    Actions disponibles (5 par défaut) :
        0 : Tout droit (gas=0.5)
        1 : Tourner gauche (steering=-0.5, gas=0.5)
        2 : Tourner droite (steering=+0.5, gas=0.5)
        3 : Accélérer (gas=1.0)
        4 : Freiner (brake=0.8)
    
    Exemple
    -------
    >>> env = gym.make("CarRacing-v2", continuous=True)
    >>> env = DiscretizeActionWrapper(env)
    >>> env.action_space
    Discrete(5)
    """

    ACTIONS = [
        np.array([0.0,  0.5, 0.0], dtype=np.float32),   # 0: Tout droit
        np.array([-0.5, 0.5, 0.0], dtype=np.float32),   # 1: Gauche léger
        np.array([0.5,  0.5, 0.0], dtype=np.float32),   # 2: Droite léger
        np.array([0.0,  1.0, 0.0], dtype=np.float32),   # 3: Accélérer
        np.array([0.0,  0.0, 0.8], dtype=np.float32),   # 4: Freiner
    ]

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def action(self, action: int) -> np.ndarray:
        return self.ACTIONS[action]


def make_env(
    render_mode: str = None,
    grayscale: bool = False,
    resize: tuple = None,
    normalize: bool = True,
    stack_frames: int = None,
    discrete: bool = False,
    seed: int = 42,
) -> gym.Env:
    """
    Factory function : crée et configure l'environnement CarRacing-v2.
    
    Paramètres
    ----------
    render_mode   : "human" pour visualiser, None pour entraînement rapide
    grayscale     : Convertir en niveaux de gris
    resize        : Tuple (H, W) pour redimensionner, ex. (42, 42)
    normalize     : Normaliser les pixels vers [0, 1]
    stack_frames  : Nombre de frames à empiler (None = désactivé)
    discrete      : Discrétiser l'espace d'action (pour DQN)
    seed          : Graine aléatoire
    
    Exemple — Env pour DQN
    ----------------------
    >>> env = make_env(
    ...     grayscale=True,
    ...     resize=(42, 42),
    ...     normalize=True,
    ...     stack_frames=4,
    ...     discrete=True,
    ... )
    
    Exemple — Env pour CNN supervisé
    ---------------------------------
    >>> env = make_env(normalize=True)
    """
    env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=True)
    
    if grayscale:
        env = GrayscaleWrapper(env)
    if resize is not None:
        env = ResizeWrapper(env, size=resize)
    if normalize:
        env = NormalizeWrapper(env)
    if stack_frames is not None:
        env = FrameStackWrapper(env, n_frames=stack_frames)
    if discrete:
        env = DiscretizeActionWrapper(env)
    
    return env
