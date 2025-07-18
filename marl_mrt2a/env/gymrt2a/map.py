"""
Map module for the robot gym environment.

This module implements a grid-based map that tracks the positions and states of robots,
objects, and obstacles in the environment. It supports communication between agents,
different observation ranges, and various action spaces.
"""

from multiprocessing.sharedctypes import Value
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import gym

from gymrt2a.robot import Robot
from gymrt2a.policy import Policy
from gymrt2a.utils import checkPosValid
from gymrt2a.object import Object

class Map:
    """
    A grid-based map for tracking robots, objects, and obstacles in the environment.
    
    This class maintains a grid representation of the environment where robots, objects,
    and obstacles are placed. It handles visibility, communication, and intention sharing
    between agents.
    
    Attributes:
        sz (Tuple[int, int]): Size of the grid
        max_range (int): Maximum observation/communication range
        comm (bool): Whether agents can communicate
        max_obj_lvl (int): Maximum object level
        one_hot_obj_lvl (bool): Whether to use one-hot encoding for object levels
        view_self (bool): Whether agents can see themselves
        action_grid (bool): Whether to use grid-based actions
        share_intention (bool): Whether agents share their intentions
    """
    
    def __init__(self, 
                 sz: Tuple[int, int], 
                 max_range: int = 5, 
                 comm: bool = False, 
                 max_obj_lvl: int = 1, 
                 one_hot_obj_lvl: bool = False, 
                 view_self: bool = True, 
                 action_grid: bool = False, 
                 share_intention: bool = False) -> None:
        """
        Initialize the map.
        
        Args:
            sz: Size of the grid (width, height)
            max_range: Maximum observation/communication range
            comm: Whether agents can communicate
            max_obj_lvl: Maximum object level
            one_hot_obj_lvl: Whether to use one-hot encoding for object levels
            view_self: Whether agents can see themselves
            action_grid: Whether to use grid-based actions
            share_intention: Whether agents share their intentions
        """
        self.sz = sz
        assert len(sz) == 2, "Invalid grid size provided"
        self.max_range = max_range
        self.lsz = (sz[0] + 2*max_range, sz[1] + 2*max_range)
        self.comm = comm
        self.max_obj_lvl = max_obj_lvl
        self.one_hot_obj_lvl = one_hot_obj_lvl
        self.view_self = view_self
        self.action_grid = action_grid
        self.share_intention = share_intention

        # self.map_dim   = (2+max_obj_lvl+(comm>0)+(share_intention is not None)) if one_hot_obj_lvl else \
        #                 (3+(comm>0)+(share_intention is not None)) ########################
        self.map_dim   = (2+max_obj_lvl+(comm>0)) if one_hot_obj_lvl else (3+(comm>0))
        # self.map_dim   = 4 if comm else 3
        # self.map_dtype = np.int64 if comm else bool
        self.map_dtype = np.float32 if share_intention else np.int64 ########################
        self._channel_robots  = 0
        self._channel_objects = [1] if not one_hot_obj_lvl else [i for i in range(1,1+max_obj_lvl)]
        # self._channel_obstacles = 2 if not one_hot_obj_lvl else 1+max_obj_lvl
        self._channel_obstacles = self._channel_objects[-1]+1
        self._channel_comm = self._channel_obstacles+1 if comm else None
        # self._channel_intention = self._channel_comm+1 if share_intention else None
        
        self.default_grid = np.full((self.map_dim,)+self.lsz, 0, dtype=self.map_dtype)
        # Add walls
        self.default_grid[self._channel_obstacles,:,:] = 1
        self.default_grid[self._channel_obstacles, max_range:-max_range, max_range:-max_range] = 0

        self.intention_grid = None
        self.joint_intention_grid = None
        
        self.grid = np.copy(self.default_grid)
        self.ind_grid = -1*np.ones_like(self.grid, dtype=int)

        self.robots: List[Robot] = []
        self.objects: List[Object] = []
        self.parser = None

    @property
    def state_space(self) -> gym.spaces.Space:
        """
        Get the state space of the environment.
        
        The state space is based on the robot's observation space but encompasses
        the whole grid. It includes channels for robots, objects, obstacles,
        communication, and visibility.
        
        Returns:
            gym.spaces.Space: The state space of the environment
        """
        # same as robot's observation, but encompassing the whole grid
        obs_space = self.robots[0].observation_space

        max_vals = self.robots[0].max_vals + 1
        max_vals = max_vals.max(axis=(1,2))  # take max val per channel
        # Add visibility channels (view and comms) at the end
        viz = (2,2) if self.comm else (2,)
        max_vals = np.append(max_vals, viz)
        max_vals = np.tile(max_vals.reshape(-1,1,1), (1,)+self.lsz)  # expand max_vals to the whole grid size
        
        if isinstance(obs_space, gym.spaces.MultiBinary):
            return gym.spaces.MultiBinary(max_vals.shape)
        elif isinstance(obs_space, gym.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(max_vals)
        elif isinstance(obs_space, gym.spaces.Box):
            return gym.spaces.Box(low=0, high=max_vals-1, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected behavior, obs_space has type: {type(obs_space)}")

    def setObstacles(self, obstacles: List[Tuple[int, int]]) -> None:
        """
        Set obstacles in the map.
        
        Args:
            obstacles: List of obstacle positions
            
        Raises:
            NotImplementedError: Obstacle collision not implemented
        """
        raise NotImplementedError("Obstacle collision not implemented")
        # Update default_grid, since obstacles are static

    def global_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert local position to global position.
        
        Args:
            pos: Local position (x, y)
            
        Returns:
            Global position (x + max_range, y + max_range)
        """
        return (pos[0] + self.max_range, pos[1] + self.max_range)

    def update(self, robots: List[Robot] = [], objs: List[Object] = []) -> None:
        """
        Update the map with new robot and object positions.
        
        Args:
            robots: List of robots to update
            objs: List of objects to update
        """
        self.grid = np.copy(self.default_grid)
        self.ind_grid = -1 * np.ones_like(self.grid, dtype=int)

        self.robots = robots
        self.objects = objs
        
        grid = np.zeros(self.lsz, dtype=self.map_dtype)
        comm_grid = np.copy(grid)
        for i in range(len(robots)):
            r = robots[i]
            x, y = self.global_pos(r.pos)
            grid[x, y] = 1  # True
            self.ind_grid[self._channel_robots, x, y] = i
            # comms
            if self.comm: 
                comm_grid[x, y] = r.comm_output

        self.grid[self._channel_robots, :, :] = grid
        if self.comm: 
            self.grid[self._channel_comm, :, :] = comm_grid

        obj_grid_sz = (len(self._channel_objects), self.lsz[0], self.lsz[1])
        grid = np.zeros(obj_grid_sz, dtype=self.map_dtype)
        for i in range(len(objs)):
            r = objs[i]
            x, y = self.global_pos(r.pos)
            if self.one_hot_obj_lvl:
                grid[r.lvl-1, x, y] = 1
                self.ind_grid[self._channel_objects[r.lvl-1], x, y] = i
            else:
                grid[0, x, y] = r.lvl
                self.ind_grid[self._channel_objects[0], x, y] = i
        self.grid[self._channel_objects, :, :] = grid

    def update_all(self) -> None:
        """
        Update all aspects of the map including communication and intention sharing.
        """
        if self.comm:               
            self.update_comm()
        if self.share_intention:    
            self.update_intention()

    def getAllObs(self) -> Tuple[np.ndarray, ...]:
        """
        Get observations for all robots.
        
        Returns:
            Tuple of observations for each robot
        """
        self.update_all()
        return tuple([self.getObs(r) for r in self.robots])

    def getObs(self, robot: Robot) -> np.ndarray:
        """
        Get observation for a specific robot.
        
        Args:
            robot: The robot to get observation for
            
        Returns:
            Observation array for the robot
            
        Raises:
            AssertionError: If observation shape is invalid
        """
        mask = robot.mask
        x, y = self.global_pos(robot.pos)
        robot_ind = self.ind_grid[self._channel_robots, x, y]
        assert robot_ind != -1
        ind_x = (x - mask.shape[1]//2, x + mask.shape[1]//2)
        ind_y = (y - mask.shape[2]//2, y + mask.shape[2]//2)
        
        if self.share_intention:
            grid = np.concatenate((self.grid, self.joint_intention_grid[robot_ind]), axis=0)
        else:
            grid = self.grid
            
        obs = grid[:, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1]
        obs = mask * obs

        if not self.view_self:
            obs[self._channel_robots, mask.shape[1]//2, mask.shape[2]//2] = 0

        # Sanity check
        assert obs.shape == robot.observation_space.shape, "Invalid observation shape"
        return obs

    def getState(self) -> np.ndarray:
        """
        Get the current state of the environment.
        
        The state includes the grid representation and visibility information.
        
        Returns:
            State array representing the current environment state
        """
        self.update_all()
        # add view and comm visibility
        if self.share_intention:
            all_intention = np.sum(self.intention_grid, axis=0)
            grid = np.concatenate((self.grid, all_intention), axis=0)
        else:
            grid = self.grid
        state = np.concatenate((grid, self.viz_grid()), axis=0)
        return state
        
    def viz_grid(self) -> np.ndarray:
        """
        Generate a visibility grid showing which places agents can see/hear.
        
        Returns:
            Boolean array indicating visible areas for each agent
        """
        viz_sz = (2,) if self.comm else (1,)
        grid = np.full(viz_sz + self.lsz, False)
        
        for r in self.robots:
            mask = r.mask
            view_mask = mask[0, :, :]
            x, y = self.global_pos(r.pos)
            ind_x = (x - mask.shape[1]//2, x + mask.shape[1]//2)
            ind_y = (y - mask.shape[2]//2, y + mask.shape[2]//2)
            
            grid[0, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1] = np.logical_or(
                grid[0, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1], 
                view_mask
            )
            
            if self.comm:
                comm_mask = mask[-1, :, :]
                grid[1, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1] = np.logical_or(
                    grid[1, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1], 
                    comm_mask
                )
        return grid

    def device_pos_grid(self) -> np.ndarray:
        """
        Generate a grid where each channel is a one-hot 2D encoding of agent positions.
        
        Returns:
            Boolean array with one-hot encoded agent positions
        """
        grid = np.full((len(self.robots),) + self.lsz, False)
        for i in range(len(self.robots)):
            r = self.robots[i]
            x, y = self.global_pos(r.pos)
            grid[i, x, y] = True
        return grid
    
    def _within_range(self, range_type: str, r1: Robot, r2: Robot) -> bool:
        """
        Check if two robots are within communication or view range of each other.
        
        Args:
            range_type: Type of range to check ("comm" or "view")
            r1: First robot
            r2: Second robot
            
        Returns:
            True if robots are within range, False otherwise
            
        Raises:
            AssertionError: If range type is invalid or ranges don't match
        """
        assert range_type in ["comm", "view"], "Invalid range type"
        range_fn = lambda r: r.comm_range if range_type == "comm" else r.view_range
        assert range_fn(r1) == range_fn(r2), "Ranges are not equal"
        r = range_fn(r1)
        dif = (abs(r1.pos[0] - r2.pos[0]), abs(r1.pos[1] - r2.pos[1]))
        return max(dif) <= r
    
    def _range_matrix(self, range_type: str, include_self: bool = False) -> np.ndarray:
        """
        Generate a matrix indicating which robots are within range of each other.
        
        Args:
            range_type: Type of range to check ("comm" or "view")
            include_self: Whether to include self-connections
            
        Returns:
            Boolean matrix where True indicates robots are within range
        """
        if include_self:
            range_matrix = np.eye(len(self.robots), dtype=bool)
        else:
            range_matrix = np.zeros((len(self.robots), len(self.robots)), dtype=bool)

        for i in range(len(self.robots)):
            for j in range(i+1, len(self.robots)):
                range_matrix[i, j] = self._within_range(range_type, self.robots[i], self.robots[j])
        range_matrix = np.logical_or(range_matrix, range_matrix.T)
        return range_matrix

    def update_comm(self) -> None:
        """
        Update the communication grid based on robot communication outputs.
        
        Raises:
            ValueError: If communication signal found without robot
        """
        for r in self.robots:
            x, y = self.global_pos(r.pos)
            self.grid[self._channel_comm, x, y] = r.comm_output
            
        # Sanity check: ensure no comm is shown if there aren't any robots
        for x in range(self.sz[0]):
            for y in range(self.sz[1]):
                gx, gy = self.global_pos((x, y))
                if not isinstance(self[x, y], Robot) and self.grid[self._channel_comm, gx, gy] > 0:
                    raise ValueError(f"A communication signal was found at {(x,y)}, but there is no robot there.")

    def update_intention(self) -> None:
        """
        Update the intention grid based on robot intentions.
        
        This method updates both individual robot intentions and joint intentions
        based on communication range.
        """
        channels = self.robots[0].intention_space().shape[0]
        # First, get all intentions and position them in a grid
        intention_grid = np.zeros((len(self.robots), channels) + self.lsz, dtype=self.map_dtype)
        for i, r in enumerate(self.robots):
            mask = r.mask
            x, y = self.global_pos(r.pos)
            ind_x = (x - mask.shape[1]//2, x + mask.shape[1]//2)
            ind_y = (y - mask.shape[2]//2, y + mask.shape[2]//2)
            intention_grid[i, :, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1] = r.intention
            
        self.intention_grid = intention_grid

        # Then, for each robot join intentions of other robots
        joint_intention_grid = np.zeros_like(intention_grid)
        comm_range_matrix = self._range_matrix("comm", include_self=False)
        for i, r in enumerate(self.robots):
            in_range = list(comm_range_matrix[i])
            joint_intention_grid[i] = np.sum(intention_grid[in_range], axis=0)

        self.joint_intention_grid = joint_intention_grid

    def available(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is available (no robots or objects).
        
        Args:
            pos: Position to check (local coordinates)
            
        Returns:
            True if position is available, False otherwise
            
        Raises:
            ValueError: If multiple objects found at same position
        """
        x, y = self.global_pos(pos)
        g = self.grid[:-1, x, y] if self.comm else self.grid[:, x, y]  # not counting the comm grid
        ind_t = np.argwhere(g)
        if ind_t.size == 0:
            return True
        elif ind_t.size == 1:
            return False
        else:
            raise ValueError("More than one object at the same position")

    def availability_grid(self) -> np.ndarray:
        """
        Generate a grid showing which positions are available.
        
        Returns:
            Boolean grid where True indicates available positions
        """
        grid = np.full(self.sz, True)
        for x in range(self.sz[0]):
            for y in range(self.sz[1]):
                grid[x, y] = self.available((x, y))
        return grid

    def available_to_move(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is available for movement.
        
        Args:
            pos: Position to check (local coordinates)
            
        Returns:
            True if position is available for movement, False otherwise
        """
        item = self.__getitem__(pos)
        if not item or isinstance(item, Object):
            return True
        else:
            return False

    def __getitem__(self, pos: Tuple[int, int]) -> Union[Robot, Object, str, bool]:
        """
        Get the item at a specific position.
        
        Args:
            pos: Position to check (local coordinates)
            
        Returns:
            Robot, Object, "obstacle", or False if empty
            
        Raises:
            AssertionError: If position is invalid
            ValueError: If multiple objects found or unexpected behavior
        """
        assert len(pos) == 2, "Invalid index"
        x, y = pos
        x, y = self.global_pos((x, y))
        g = self.grid[:-1, x, y] if self.comm else self.grid[:, x, y]  # not counting the comm grid
        ind_t = np.argwhere(g)
        if ind_t.size == 0:
            return False
        elif ind_t.size > 1:
            raise ValueError("More than one object at the same position")

        ind_t = ind_t.item()
        if ind_t == self._channel_robots:
            return self.robots[self.ind_grid[ind_t, x, y]]
        elif ind_t == self._channel_obstacles:
            return "obstacle"
        elif ind_t in self._channel_objects:
            return self.objects[self.ind_grid[ind_t, x, y]]
        else:
            raise ValueError(f"Unexpected behavior, __getitem__, args: {pos}")

    def pop(self, pos: Tuple[int, int]) -> Union[Robot, Object, str]:
        """
        Remove and return the item at a specific position.
        
        Args:
            pos: Position to pop from (local coordinates)
            
        Returns:
            The removed item (Robot, Object, or "obstacle")
            
        Raises:
            AssertionError: If position is invalid
            ValueError: If nothing to pop or multiple objects found
        """
        assert len(pos) == 2, "Invalid index"
        x, y = self.global_pos(pos)
        g = self.grid[:-1, x, y] if self.comm else self.grid[:, x, y]  # not counting the comm grid
        ind_t = np.argwhere(g)
        if ind_t.size == 0:
            return None
        elif ind_t.size > 1:
            raise RuntimeError(f"More than one object at the same position, pos={pos}")

        ind_t = ind_t.item()
        if ind_t == self._channel_robots:
            old = self.robots.pop(self.ind_grid[ind_t, x, y])
            for i in range(self.ind_grid[ind_t, x, y], len(self.robots)):
                r = self.robots[i]
                pos = self.global_pos(r.pos)
                assert self.ind_grid[self._channel_robots, pos[0], pos[1]] >= 0, "!1"
                self.ind_grid[self._channel_robots, pos[0], pos[1]] -= 1
        elif ind_t in self._channel_objects:
            old = self.objects.pop(self.ind_grid[ind_t, x, y])
            for i in range(self.ind_grid[ind_t, x, y], len(self.objects)):
                r = self.objects[i]
                pos = self.global_pos(r.pos)
                if self.one_hot_obj_lvl:
                    assert self.ind_grid[self._channel_objects[r.lvl-1], pos[0], pos[1]] >= 0, "!2"
                    self.ind_grid[self._channel_objects[r.lvl-1], pos[0], pos[1]] -= 1
                else:
                    assert self.ind_grid[self._channel_objects[0], pos[0], pos[1]] >= 0, "!3"
                    self.ind_grid[self._channel_objects[0], pos[0], pos[1]] -= 1
        elif ind_t == self._channel_obstacles:
            old = "obstacle"
        else:
            raise ValueError(f"Unexpected behavior, pop, pos: {pos}")
            
        self.grid[:, x, y] = 0  # False
        self.ind_grid[:, x, y] = -1
        return old

    def remove(self, val: Union[Robot, Object]) -> bool:
        """
        Remove a robot or object from the map.
        
        Args:
            val: Robot or object to remove
            
        Returns:
            True if item was removed, False otherwise
        """
        for r in self.robots:
            if r is val:
                pos = val.pos
                self.pop(pos)
                return True
        for r in self.objects:
            if r is val:
                pos = val.pos
                self.pop(pos)
                return True
        return False

    def __setitem__(self, pos: Tuple[int, int], val: Union[Robot, Object, str, None]) -> bool:
        """
        Set an item at a specific position.
        
        Args:
            pos: Position to set (local coordinates)
            val: Item to set (Robot, Object, "obstacle", or None)
            
        Returns:
            True if item was set successfully
            
        Raises:
            AssertionError: If position is invalid
            ValueError: If value type is unexpected
        """
        assert len(pos) == 2, "Invalid index"
        x, y = self.global_pos(pos)

        # If item is already in the map, move it instead
        if val in self.robots or val in self.objects:
            return self.move(val.pos, pos)

        # Delete old item
        old = self.pop(pos)
        
        if isinstance(val, Robot):
            self.robots.append(val)
            self.grid[self._channel_robots, x, y] = 1  # True
            if self.comm: 
                self.grid[self._channel_comm, x, y] = val.comm_output
            self.ind_grid[self._channel_robots, x, y] = len(self.robots) - 1
        elif isinstance(val, Object):
            self.objects.append(val)
            ind = self._channel_objects[0] if not self.one_hot_obj_lvl else self._channel_objects[val.lvl-1]
            self.grid[ind, x, y] = 1 if self.one_hot_obj_lvl else val.lvl
            self.ind_grid[ind, x, y] = len(self.objects) - 1
        elif val == "obstacle":
            self.grid[self._channel_obstacles, x, y] = 1  # True
        elif val is None:
            self.pop(pos)
        else:
            raise ValueError("Unexpected value type")
            
        val.pos = pos
        return True
        
    def move(self, pos: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
        """
        Move an item from one position to another.
        
        Args:
            pos: Current position (local coordinates)
            next_pos: Target position (local coordinates)
            
        Returns:
            True if move was successful
            
        Raises:
            AssertionError: If positions are invalid
            ValueError: If destination is unavailable or nothing to move
        """
        assert len(pos) == 2, "Invalid index"
        assert len(next_pos) == 2, "Invalid index"
        assert checkPosValid(pos, self.sz), f"Invalid index: {pos}, size: {self.sz}"
        assert checkPosValid(next_pos, self.sz), f"Invalid index: {pos}, size: {self.sz}"

        # Check if destination is available
        if self.available_to_move(next_pos):
            val = self[pos]
            assert val, f"There's nothing to move at {pos}"

            x, y = self.global_pos(pos)
            self.grid[:, x, y] = 0
            self.ind_grid[:, x, y] = -1
            
            old = self.pop(next_pos)
            x, y = self.global_pos(next_pos)
            if isinstance(val, Robot):
                self.grid[self._channel_robots, x, y] = 1
                if self.comm: 
                    self.grid[self._channel_comm, x, y] = val.comm_output
                # Find current index of robot
                val_ind = self.robots.index(val)
                self.ind_grid[self._channel_robots, x, y] = val_ind
            elif isinstance(val, Object):
                ind = self._channel_objects[0] if not self.one_hot_obj_lvl else self._channel_objects[val.lvl-1]
                self.grid[ind, x, y] = 1 if self.one_hot_obj_lvl else val.lvl
                # Find current index of robot
                val_ind = self.objects.index(val)
                self.ind_grid[ind, x, y] = val_ind
            elif val == "obstacle":
                self.grid[self._channel_obstacles, x, y] = 1
            else:
                raise ValueError("Unexpected value type")
            val.pos = next_pos
            return True
        else:
            raise ValueError(f"Tried to move to unavailable position, next_pos={next_pos}")

    def get_channel_info(self) -> Dict[str, Any]:
        """
        Get information about the different channels in the map.
        
        Returns:
            Dictionary containing channel information
        """
        assert len(self.robots) > 0, "Tried to get channel info before setting robots"
        if self.share_intention:
            intention_first_channel = self._channel_obstacles + 1 if self._channel_comm is None \
                                                                else self._channel_comm + 1
            n_intention_channels = self.robots[0].intention_space().shape[0]
            intention_channels = list(range(intention_first_channel, intention_first_channel + n_intention_channels))
        else:
            intention_channels = None

        return {
            "robots": self._channel_robots,
            "objects": self._channel_objects,
            "obstacles": self._channel_obstacles,
            "comm": self._channel_comm,
            "intention": intention_channels,
        }