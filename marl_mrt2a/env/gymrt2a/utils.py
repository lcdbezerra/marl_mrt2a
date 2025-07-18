"""Utility functions for the robot gym environment.

This module provides various helper functions for the robot gym environment,
including observation mask creation, action processing, and environment utility functions.
"""

from __future__ import annotations
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
import gym

import gymrt2a
from gymrt2a.controller import act as action_dict


def is_ibex() -> bool:
    """Check if running on IBEX compute cluster.
    
    Returns:
        True if running on IBEX, False otherwise
    """
    if os.getenv("KAUST_ARCH") is None:
        return False
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    return True


def createObsMask(r: int, max_r: Optional[int] = None) -> np.ndarray:
    """Create a square observation mask.
    
    Args:
        r: Radius of the observation mask
        max_r: Maximum radius for padding
        
    Returns:
        Boolean mask array with True values in a square
    """
    L = 2 * r + 1
    mat = np.full((L, L), True, dtype=bool)
    if max_r and max_r > r:
        mat = np.pad(mat, max_r - r)
    return mat


def createCircularObsMask(r: int, max_r: Optional[int] = None) -> np.ndarray:
    """Create a circular observation mask.
    
    Args:
        r: Radius of the observation mask
        max_r: Maximum radius for padding
        
    Returns:
        Boolean mask array with True values in a circle
    """
    L = 2 * r + 1
    R = L / 2
    mask = np.full((L, L), False, dtype=bool)

    for x in range(L):
        for y in range(L):
            mask[x, y] = ((x + 0.5 - L / 2) ** 2 + (y + 0.5 - L / 2) ** 2) <= ((L / 2) ** 2)
            
    if max_r and max_r > r:
        mask = np.pad(mask, max_r - r)
    return mask


def getNextPosFromAction(pos: Tuple[int, int], act: int) -> Tuple[int, int]:
    """Get the next position after taking an action.
    
    Args:
        pos: Current position (x, y)
        act: Action to take
        
    Returns:
        New position (x, y) after taking the action
        
    Raises:
        ValueError: If the action is unknown
    """
    if act == action_dict["0"]:
        return pos
    elif act == action_dict["U"]:
        return (pos[0], pos[1] + 1)
    elif act == action_dict["D"]:
        return (pos[0], pos[1] - 1)
    elif act == action_dict["L"]:
        return (pos[0] - 1, pos[1])
    elif act == action_dict["R"]:
        return (pos[0] + 1, pos[1])
    elif act == action_dict["UR"]:
        return (pos[0] + 1, pos[1] + 1)
    elif act == action_dict["DR"]:
        return (pos[0] + 1, pos[1] - 1)
    elif act == action_dict["DL"]:
        return (pos[0] - 1, pos[1] - 1)
    elif act == action_dict["UL"]:
        return (pos[0] - 1, pos[1] + 1)
    else:
        raise ValueError(f"Unknown action provided: {act}")


def checkPosValid(pos: Tuple[int, int], sz: Tuple[int, int]) -> bool:
    """Check if a position is within grid bounds.
    
    Args:
        pos: Position to check (x, y)
        sz: Grid size (width, height)
        
    Returns:
        True if position is valid, False otherwise
    """
    if pos[0] < 0 or pos[1] < 0:
        return False
    elif pos[0] >= sz[0]:
        return False
    elif pos[1] >= sz[1]:
        return False
    return True


def xy_print(g: np.ndarray) -> None:
    """Print a grid with x,y coordinates.
    
    Args:
        g: Grid to print
    """
    for y in range(g.shape[1]):
        print("[", end=" ")
        for x in range(g.shape[0]):
            print(g[x, y], end=" ")
        print(" ]")


def binmat_print(g: np.ndarray) -> None:
    """Print a binary matrix as a visual grid.
    
    Args:
        g: Binary matrix to print
    """
    for y in range(g.shape[1] - 1, -1, -1):
        print("[", end="")
        for x in range(g.shape[0]):
            print("-" if g[x, y] else " ", end="")
        print("]")


def printAction(act: int) -> None:
    """Print the string representation of an action.
    
    Args:
        act: Action code to print
    """
    keys = list(action_dict.keys())
    ind = list(action_dict.values())
    print(keys[ind.index(act)])


def sepAction(act: int, sizes: List[int]) -> Tuple[int, int]:
    """Separate navigation and communication actions.
    
    Args:
        act: Combined action code
        sizes: List of action space sizes [nav_size, comm_size]
        
    Returns:
        Tuple of (nav_action, comm_action)
        
    Raises:
        AssertionError: If action or sizes are invalid
    """
    assert act < np.prod(sizes), f"Invalid action: {act}"
    assert act == int(act), f"Invalid action, expected integer: {act}"
    assert len(sizes) == 2, f"Invalid size array: {sizes}"

    nav_act = act % sizes[1]
    comm_act = act // sizes[1]
    
    return nav_act, comm_act


class ObservationParser:
    """Parser for agent observations in the environment.
    
    This class helps extract different components from an agent's observation,
    such as robot positions, objects, and obstacles.
    """
    
    def __init__(self, r: Any):
        """Initialize the observation parser.
        
        Args:
            r: Reference to the environment or agent
        """
        self.r = r
        self.action_space = r.action_space
        self.observation_space = r.observation_space
        self.comm = r.comm
        self.share_intention = r.share_intention

        space_max = (r.robot_space(), r.object_space(), r.obstacle_space())
        self.robot_channels = [i for i in range(space_max[0].shape[0])]
        curr_ind = self.robot_channels[-1]
        self.object_channels = [curr_ind + 1 + i for i in range(space_max[1].shape[0])]
        curr_ind = self.object_channels[-1]
        self.obstacle_channels = [curr_ind + 1 + i for i in range(space_max[2].shape[0])]
        curr_ind = self.obstacle_channels[-1]

        if self.comm: 
            space_max += (r.comm_space(),)
            self.comm_channels = [curr_ind + 1 + i for i in range(space_max[-1].shape[0])]
            curr_ind = self.comm_channels[-1]
        else:
            self.comm_channels = []

        if self.share_intention:
            space_max += (r.intention_space(),)
            self.intention_channels = [curr_ind + 1 + i for i in range(space_max[-1].shape[0])]
            curr_ind = self.intention_channels[-1]
        else:
            self.intention_channels = []

    def __call__(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Parse an observation into its components.
        
        Args:
            obs: Raw observation array
            
        Returns:
            Dictionary with parsed observation components
        """
        return {
            "robot": obs[self.robot_channels, :, :],
            "object": obs[self.object_channels, :, :],
            "obstacle": obs[self.obstacle_channels, :, :],
            "comm": obs[self.comm_channels, :, :] if self.comm else None,
            "intention": obs[self.intention_channels, :, :] if self.share_intention else None,
        }
    
    def not_passable_grid(self, obs: np.ndarray) -> np.ndarray:
        """Create a grid showing positions that are not passable.
        
        Args:
            obs: Raw observation array
            
        Returns:
            Boolean array where True indicates a non-passable position
        """
        parsed = self(obs)
        robot_not_passable = parsed["robot"]
        not_passable = np.concatenate(
            (robot_not_passable, parsed["object"][1:, :, :], parsed["obstacle"]), 
            axis=0
        )
        not_passable = np.logical_or.reduce(not_passable, axis=0)
        return not_passable
    
    def _filter_adjacent(self, obs: np.ndarray) -> np.ndarray:
        """Filter observation to only include adjacent positions.
        
        Args:
            obs: Observation array to filter
            
        Returns:
            Filtered observation array
        """
        sz = obs.shape[-2:]
        center = sz[0] // 2
        mask = np.zeros((1, *sz))
        mask[0, [center-1, center, center+1], [center-1, center, center+1]] = 1
        return obs * mask


###########################################################################################################################
# From MPE (https://github.com/semitable/multiagent-particle-envs)

# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.RandomState().rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    mask = createCircularObsMask(2, 3)
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    print(mask)
    plt.show()