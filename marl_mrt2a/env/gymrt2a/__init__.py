# Standard library imports
from itertools import product

# Third-party imports
from gym.envs.registration import register

# Local imports
from . import env
# from . import consensus_env
from . import env_config
from . import env_recorder

# Core components
from .robot import GymRobot
from .object import Object

# Utilities and wrappers
from .utils import *
from .wrapper import *