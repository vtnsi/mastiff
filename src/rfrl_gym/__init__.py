

__version__ = '0.2.0.alpha'

import numpy as np
from gymnasium.envs.registration import register

import warnings
import sys
if sys.version_info >= (3,11):
    with warnings.catch_warnings(action="ignore"):
        register(
            id='rfrl-gym-abstract-v0',
            entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
            max_episode_steps = np.inf
        )
        register(
            id='rfrl-gym-iq-v0',
            entry_point='rfrl_gym.envs:RFRLGymIQEnv',
            max_episode_steps = np.inf
        )
        register(
            id='rfrl-gym-adversarial-v0',
            entry_point='rfrl_gym.envs:RFRLGymAdversarialEnv',
            max_episode_steps = np.inf
        )
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        register(
            id='rfrl-gym-abstract-v0',
            entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
            max_episode_steps = np.inf
        )
        register(
            id='rfrl-gym-iq-v0',
            entry_point='rfrl_gym.envs:RFRLGymIQEnv',
            max_episode_steps = np.inf
        )
        register(
            id='rfrl-gym-adversarial-v0',
            entry_point='rfrl_gym.envs:RFRLGymAdversarialEnv',
            max_episode_steps = np.inf
        )


from . import entities
from . import envs
from . import renderers
from . import signals
from . import spectrums

__all__ = [
    "entities",
    "envs",
    "renderers",
    "signals",
    "spectrums"
]

del sys,np,warnings,register