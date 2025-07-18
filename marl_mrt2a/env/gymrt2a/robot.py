"""
Robot implementations for robot gym environments.

This module provides robot classes that can be used in gym-based robot simulation
environments, with support for observation processing, action handling, and intention sharing.
"""

from gymrt2a.utils import *
from gymrt2a.policy import *
from gymrt2a.controller import *

import numpy as np
import gym

class Robot:
    pass


class GymRobot(Robot):
    """
    Robot agent implementation for gym-based environments.
    
    Handles observations, actions, and optional communication between agents.
    Supports various observation spaces and intention sharing mechanisms.
    
    Attributes:
        pos: Current position of the robot
        view_range: Field of view range
        comm_range: Communication range
        comm: Communication dimension (0 if no communication)
        action_grid: Whether to use grid-based actions
        share_intention: Type of intention sharing (None, "q_grid", "target", "path", "channel")
        max_obj_lvl: Maximum object level in the environment
        one_hot_obj_lvl: Whether to use one-hot encoding for object levels
    """
    def __init__(self, 
                 start_pos, 
                 view_range=1, 
                 comm_range=None, 
                 comm=False, 
                 hardcoded=False,
                 max_obj_lvl=1, 
                 one_hot_obj_lvl=False, 
                 circular_obs_mask=False, 
                 controller=None, 
                 action_grid=False, 
                 share_intention=False,
                 intention_steps=0):
        """
        Initialize a GymRobot.
        
        Args:
            start_pos: Initial position (x, y)
            view_range: Field of view range
            comm_range: Communication range (defaults to view_range if None)
            comm: Communication dimension (False if no communication)
            hardcoded: Control mode ('comm', 'nav', True, or False)
            max_obj_lvl: Maximum object level in the environment
            one_hot_obj_lvl: Whether to use one-hot encoding for object levels
            circular_obs_mask: Whether to use circular observation mask
            controller: Controller class to use
            action_grid: Whether to use grid-based actions
            share_intention: Type of intention sharing (False or one of: "q_grid", "target", "path", "channel")
            intention_steps: Number of steps for intention sharing
        """
        self.comm = comm
        self.view_range = view_range
        self.comm_range = comm_range if comm_range is not None else view_range
        self._mask_fn = createCircularObsMask if circular_obs_mask else createObsMask
        self._max_range = max(self.view_range, self.comm_range)
        self._view_mask = self._mask_fn(self.view_range, self._max_range)
        self._comm_mask = self._mask_fn(self.comm_range, self._max_range)
        self.action_grid = action_grid
        self.share_intention = share_intention
        self.intention_steps = self._max_range if (self.share_intention == "channel" and intention_steps == 0) \
                                                else intention_steps
        self._intention = None
        self._comm_output = 0
        self.max_obj_lvl = max_obj_lvl
        self.one_hot_obj_lvl = one_hot_obj_lvl

        self.last_obs = None
        self.hardcoded = hardcoded

        if comm:
            assert comm == int(comm), "Invalid agent communication dimension provided"
        assert max_obj_lvl == int(max_obj_lvl) and max_obj_lvl >= 1, "Invalid max_obj_lvl provided"

        self.pos = start_pos
        self.command = None

        space_max = (self.robot_space(), self.object_space(), self.obstacle_space())
        if self.comm: 
            space_max += (self.comm_space(),)
        if not self.share_intention:
            space_max = np.concatenate(space_max, axis=0)
            if np.max(space_max) == 2:
                self.observation_space = gym.spaces.MultiBinary(space_max.shape)
            else:
                self.observation_space = gym.spaces.MultiDiscrete(space_max)
            self.max_vals = space_max - 1
        else:           
            space_max += (self.intention_space(),)
            space_max = np.concatenate(space_max, axis=0) - 1
            self.max_vals = space_max
            self.observation_space = gym.spaces.Box(low=0, high=space_max, dtype=np.float32)
        self.action_space = self._action_space()

        self._build_mask()

        self.parser = ObservationParser(self)
        self.controller = (lambda *args, **kw: None) if controller is None else (controller(self))
    
    def _build_mask(self):
        """Build the observation mask for the robot."""
        mask = []
        for _ in range(self.robot_space().shape[0]):    
            mask.append(np.expand_dims(self._view_mask, 0))
        for _ in range(self.object_space().shape[0]):   
            mask.append(np.expand_dims(self._view_mask, 0))
        for _ in range(self.obstacle_space().shape[0]): 
            mask.append(np.expand_dims(self._view_mask, 0))
        
        s = self.comm_space()
        if s is not None: 
            for _ in range(s.shape[0]): 
                mask.append(np.expand_dims(self._comm_mask, 0))
        
        s = self.intention_space()
        if s is not None:
            for _ in range(s.shape[0]): 
                mask.append(np.expand_dims(self._comm_mask, 0))
        
        mask = np.concatenate(mask, axis=0)
        self.mask = mask

    def robot_space(self):
        """Define the robot observation space."""
        mat = 2 * np.expand_dims(np.ones_like(self._view_mask), 0)
        return mat

    def object_space(self):
        """Define the object observation space."""
        if not self.one_hot_obj_lvl:
            mat = (self.max_obj_lvl + 1) * np.expand_dims(np.ones_like(self._view_mask), 0)
        else:
            mat = np.expand_dims(2 * np.ones_like(self._view_mask), 0)
            mat = np.tile(mat, (self.max_obj_lvl, 1, 1))
        return mat

    def obstacle_space(self):
        """Define the obstacle observation space."""
        mat = 2 * np.expand_dims(np.ones_like(self._view_mask), 0)
        return mat
    
    def comm_space(self):
        """Define the communication observation space."""
        if not self.comm: 
            return None
        mat = self.comm * np.expand_dims(np.ones_like(self._comm_mask), 0)
        return mat

    def intention_space(self):
        """Define the intention observation space."""
        if not self.share_intention: 
            return None
        mat = np.expand_dims(np.full_like(self._view_mask, np.inf, dtype=np.float32), 0)
        if self.share_intention == "channel":
            mat = np.tile(mat, (self.intention_steps, 1, 1))
        return mat
        
    def _action_space(self):
        """Define the action space for the robot."""
        return gym.spaces.Tuple((self._nav_action_space(), self._comm_action_space()))
    
    def _nav_action_space(self):
        """Define the navigation action space."""
        if self.action_grid:
            # Grid-based action space
            return gym.spaces.Tuple((
                gym.spaces.Discrete(2 * self._max_range + 1),
                gym.spaces.Discrete(2 * self._max_range + 1)
            ))
        else:
            # Steering action space
            return gym.spaces.Discrete(9)
    
    def _comm_action_space(self):
        """Define the communication action space."""
        if self.comm >= 2:
            return gym.spaces.Discrete(self.comm)
        else:
            return gym.spaces.Discrete(1)  # always silent
        
    @property
    def comm_output(self):
        """Get the current communication output value."""
        if not self.comm:
            raise ValueError("Robot has no communication ability")
        return self._comm_output
    
    @comm_output.setter
    def comm_output(self, value):
        """Set the communication output value."""
        if not self.comm:
            raise ValueError("Robot has no communication ability")
        elif not value == int(value) or value >= self.comm or value < 0:
            raise ValueError(f"Invalid communication signal provided: {value}")
        self._comm_output = value

    def process_action(self, act, next_step=True):
        """
        Process an action tuple and convert it to controller commands.
        
        Args:
            act: Tuple of (navigation command, communication signal)
            next_step: Whether to advance the controller state
            
        Returns:
            Navigation action
        """
        nav_cmd, comm = act
        if self.comm:
            self.comm_output = comm
        self.controller.assign_command(nav_cmd, self.pos)
        if self.last_obs is not None:
            nav_act = self.controller(self.last_obs, self.pos, next_step=next_step)
        else:
            nav_act = 0
        return nav_act
    
    def process_obs(self, obs):
        """
        Process and store an observation.
        
        Args:
            obs: The observation to process
            
        Returns:
            True if processing was successful
        """
        self.last_obs = obs
        return True
        
    @property
    def intention(self):
        """Get the current intention representation based on sharing type."""
        if not self.share_intention:
            return None
        elif self.share_intention == "q_grid":
            return self._intention if self._intention is not None else np.zeros_like(self._view_mask)
        elif self.share_intention == "target":
            path = self.controller.path
            return self.path_to_target(path)
        elif self.share_intention == "path":
            path = self.controller.path
            return self.path_to_grid(path)
        elif self.share_intention == "channel":
            path = self.controller.path
            return self.path_to_channels(path)
        else:
            raise ValueError(f"Unexpected behavior, share_intention='{self.share_intention}'")
        
    @intention.setter
    def intention(self, arg):
        """
        Set the intention for q_grid sharing mode.
        
        Args:
            arg: The intention grid
            
        Returns:
            True if intention was set successfully
        """
        if self.share_intention in [None, "target", "path", "channel"]:
            return False
        elif self.share_intention == "q_grid":
            assert arg.shape == self._view_mask.shape, f"Invalid intention shape: {arg.shape}"
            self._intention = arg
            return True
        else:
            raise ValueError(f"Unexpected behavior, share_intention='{self.share_intention}'")
    
    def path_to_target(self, path):
        """
        Convert a path to a target-based intention representation.
        
        Args:
            path: List of path points
            
        Returns:
            Grid with target point marked
        """
        sz = self.mask.shape[1:]
        grid = np.zeros((1, *sz))
        if path is None: 
            return grid
        
        path_to_print = path if not self.intention_steps else path[:self.intention_steps]
        ind = (path_to_print[-1][0] - self.pos[0] + sz[0] // 2, 
               path_to_print[-1][1] - self.pos[1] + sz[1] // 2)
        try:
            grid[0, ind[0], ind[1]] = 1
        except IndexError:
            pass

        return grid

    def path_to_grid(self, path, decay=0.67):
        """
        Convert a path to a grid-based intention representation with decay.
        
        Args:
            path: List of path points
            decay: Decay factor for path points
            
        Returns:
            Grid with decaying path values
        """
        sz = self.mask.shape[1:]
        grid = np.zeros((1, *sz))
        if path is None: 
            return grid
        
        path_to_print = path if not self.intention_steps else path[:self.intention_steps]
        inds = [(p[0] - self.pos[0] + sz[0] // 2, 
                 p[1] - self.pos[1] + sz[1] // 2) for p in path_to_print]
        
        for i, (x, y) in enumerate(inds):
            try:
                grid[0, x, y] = decay ** (i + 1)
            except IndexError:
                break
                
        return grid

    def path_to_channels(self, path):
        """
        Convert a path to a channel-based intention representation.
        
        Args:
            path: List of path points
            
        Returns:
            Multi-channel grid with path points
        """
        sz = self.mask.shape[1:]
        grid = np.zeros((self.intention_steps, *sz))
        if path is None: 
            return grid
        
        path_to_print = path[:self.intention_steps]
        inds = [(p[0] - self.pos[0] + sz[0] // 2, 
                 p[1] - self.pos[1] + sz[1] // 2) for p in path_to_print]
        
        for i, (x, y) in enumerate(inds):
            try:
                grid[i, x, y] = 1
            except IndexError:
                break
                
        return grid

if __name__=="__main__":
    r = GymRobot(start_pos=(0,0),
                 view_range=2,
                 comm_range=5,
                 max_obj_lvl=3, 
                 one_hot_obj_lvl=True, 
                 comm=0,
                 circular_obs_mask=True,
                 controller=PathfindingController,
                 action_grid=True,
                 share_intention=True,)

    print(r.observation_space)
    print(r.action_space)
    print(r.mask)
    # for i in range(r.mask.shape[-1]): print(r.mask[:,:,i])
    # print(r)

    # sz = (17,17)
    sz = (r.mask.shape[0], r.mask.shape[1])
    act = np.zeros(sz, dtype=np.int8)
    ind = (np.random.randint(sz[0]), np.random.randint(sz[1]))
    act[ind] = 1
    print(act)
    print(ind)

    # flatten using torch
    import torch
    a = torch.flatten(torch.Tensor(act))

    pred_ind = torch.argmax(a)
    print(pred_ind)

    r.process_action(pred_ind.item())


    pred_ind = np.unravel_index(pred_ind, sz)
    print(pred_ind)

    print("test")