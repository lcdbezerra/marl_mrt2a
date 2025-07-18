import numpy as np
from math import ceil
from typing import Tuple, List, Dict, Any, Optional, Union, Callable


class Policy:
    """Base policy class for robot agents.
    
    This class defines the interface for all policies and provides utility methods
    for common operations.
    """
    
    def __init__(self, robot):
        """Initialize policy with a robot instance.
        
        Args:
            robot: Robot instance this policy controls
        """
        self._robot = robot
        self.parser = robot.parser

    def __call__(self, obs) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Execute policy based on observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (navigation_command, communication_command)
        """
        raise NotImplementedError("Base Policy")

    # Utility methods
    def _get_closest_ind(self, grid) -> Tuple[Optional[int], Optional[int]]:
        """Find closest object in grid relative to center.
        
        Args:
            grid: 2D grid of observations
            
        Returns:
            Relative coordinates (y, x) of closest object or (None, None) if grid is empty
        """
        ind = np.argwhere(grid)
        if ind.size == 0:
            # No objects found, return None
            return None, None
            
        # Find closest point
        center = grid.shape[0] // 2
        rel_ind = ind - center
        rel_abs_ind = np.sum(np.abs(rel_ind), axis=1)
        i = np.argmin(rel_abs_ind)
        # Returns RELATIVE coordinates of desired destination
        return tuple(rel_ind[i, :])
    
    def _update_nav_cmd(self, 
                         robot, 
                         old_pos: Tuple[int, int], 
                         nav_cmd: Optional[Tuple[int, int]], 
                         obs: Dict[str, np.ndarray]) -> Optional[Tuple[int, int]]:
        """Update navigation command based on robot movement.
        
        Args:
            robot: Robot instance
            old_pos: Previous robot position
            nav_cmd: Previous navigation command
            obs: Current observation
            
        Returns:
            Updated navigation command or None if target no longer valid
        """
        # Update command according to agent's movement
        if nav_cmd is None or nav_cmd == (None, None): 
            return None
            
        robot_dif = (robot.pos[0] - old_pos[0], robot.pos[1] - old_pos[1])
        nav_cmd = (nav_cmd[0] - robot_dif[0], nav_cmd[1] - robot_dif[1])
        center = obs["object"].shape[1] // 2
        
        # Check if there is an object at the target
        try:
            if obs["object"][:, nav_cmd[0] + center, nav_cmd[1] + center].any():
                return nav_cmd
            else:
                return None
        except IndexError:
            return None
    

class PolicyArray:
    """Manages a collection of policies for multiple robots."""
    
    def __init__(self, policy_fn: Callable, robots=None, policy_kwargs: Dict[str, Any] = {}):
        """Initialize policy array.
        
        Args:
            policy_fn: Function that creates a policy instance
            robots: List of robot instances to create policies for
            policy_kwargs: Keyword arguments to pass to policy_fn
        """
        self.policy_fn = policy_fn
        self.policies = None
        self.policy_kwargs = policy_kwargs
        if robots is not None:
            self.reset(robots)

    def reset(self, robots: List) -> None:
        """Reset policies with new robot instances.
        
        Args:
            robots: List of robot instances
        """
        self.policies = [self.policy_fn(r, **self.policy_kwargs) for r in robots]
    
    def __call__(self, obs: List) -> List:
        """Execute all policies with corresponding observations.
        
        Args:
            obs: List of observations for each robot
            
        Returns:
            List of actions for each robot
        """
        actions = [self.policies[i](o) for i, o in enumerate(obs)]
        return actions
        

###############################################################
# Communication Policies
###############################################################

class CommunicationPolicy(Policy):
    """Base class for policies that can communicate."""
    
    def __init__(self, robot):
        """Initialize communication policy.
        
        Args:
            robot: Robot instance with communication capabilities
        """
        super().__init__(robot)
        self.comm = robot.comm
        if not self.comm: 
            raise RuntimeWarning("Communication policy initialized, but no communication is available")


class HighestObjLvlCommPolicy(CommunicationPolicy):
    """Communicates the highest object level observed.
    
    When called, returns the highest object level it sees, or zero if no objects around.
    """
    
    def __init__(self, robot):
        """Initialize policy.
        
        Args:
            robot: Robot instance with one-hot object level encoding
        """
        super().__init__(robot)
        assert robot.one_hot_obj_lvl, "Supports only one hot object level for now"

    def __call__(self, obs) -> Tuple[None, int]:
        """Execute policy based on observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (None, communication_command)
        """
        obs = self.parser(obs)
        obj_inds = list(range(self._robot.max_obj_lvl))[::-1]  # Descending order
        
        for i in obj_inds:  # From highest level to lowest
            obj_grid = obs["object"][i, :, :]
            ind = np.argwhere(obj_grid)
            if ind.size > 0:
                return None, min(i, self.comm - 1)
        
        # If no object is found in the surroundings
        return None, 0


class TargetObjLvlCommPolicy(CommunicationPolicy):
    """Communicates the target object level.
    
    When called, returns its target object level, or zero if no object at the target.
    """
    
    def __init__(self, robot):
        """Initialize policy.
        
        Args:
            robot: Robot instance with one-hot object level encoding
        """
        super().__init__(robot)
        assert robot.one_hot_obj_lvl, "Supports only one hot object level for now"
        self.view_range = robot.view_range

    def __call__(self, obs) -> int:
        """Execute policy based on observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Communication command indicating object level
        """
        obs = self.parser(obs)
        cmd = self._robot.controller.cmd
        if cmd is None or cmd == (None, None):
            return 0
            
        pos = self._robot.pos
        dif = (cmd[0] - pos[0], cmd[1] - pos[1])
        
        # Check if target within view range
        if abs(dif[0]) > self.view_range or abs(dif[1]) > self.view_range:
            return 0
            
        sz = 2 * self.view_range + 1
        dif = (dif[0] + sz // 2, dif[1] + sz // 2)
        objs = obs["object"][:, dif[0], dif[1]]
        ind = np.argwhere(objs)
        
        assert ind.size in [0, 1], "Multiple object levels at the same position"
        
        if ind.size > 0:
            return int(ind[0]) + 1
        else:
            return 0

    def _dif_within_obs(self, dif: Tuple[int, int]) -> bool:
        """Check if difference is within observation bounds.
        
        Args:
            dif: Difference to check
            
        Returns:
            True if difference is within bounds
        """
        sz = self.out_shape[1:]
        return (0 <= dif[0] + sz[0] // 2 < sz[0] and 
                0 <= dif[1] + sz[1] // 2 < sz[1])


###############################################################
# Navigation Policies
###############################################################

class NavigationPolicy(Policy):
    """Base class for navigation-only policies."""
    pass


class RandomPointPolicy(NavigationPolicy):
    """Policy that navigates to random points."""
    
    def __call__(self, obs) -> Tuple[Tuple[int, int], None]:
        """Execute policy to generate a random target.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (navigation_command, None)
        """
        command = (np.random.randint(self._robot.sz[0]), 
                   np.random.randint(self._robot.sz[1]))
        return command, None


###############################################################
# Joint Navigation+Communication Policies
###############################################################

class JointPolicy(Policy):
    """Base class for policies that handle both navigation and communication."""
    
    def __init__(self, robot):
        """Initialize joint policy.
        
        Args:
            robot: Robot instance
        """
        super().__init__(robot)
        self.comm = robot.comm


class NonePolicy(JointPolicy):
    """Policy that does nothing."""
    
    def __call__(self, obs) -> Tuple[None, None]:
        """Execute policy that takes no action.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (None, None)
        """
        return None, None


class ClosestObjPolicy(JointPolicy):
    """Policy that navigates to closest object."""
    
    def __call__(self, obs) -> Tuple[Tuple[Optional[int], Optional[int]], None]:
        """Execute policy to find closest object.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (navigation_command, None)
        """
        obj_grid = obs[1, :, :]
        return self._get_closest_ind(obj_grid), None


class HighestLvlObjPolicy(ClosestObjPolicy):
    """Policy that prioritizes highest level objects.
    
    Navigates to the closest highest-level object in view.
    """
    
    def __init__(self, robot, agent_reeval_rate=True):
        """Initialize policy.
        
        Args:
            robot: Robot instance with one-hot object level encoding
            agent_reeval_rate: Probability of reevaluating target each step
        """
        super().__init__(robot)
        assert robot.one_hot_obj_lvl, "Supports only one hot object level for now"
        self.reeval_prob = 1.0 if agent_reeval_rate is True else (1 - np.exp(-agent_reeval_rate))

        self.nav_cmd = None
        self.comm_cmd = None
        self.robot_pos = robot.pos

    def __call__(self, obs) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Execute policy to find highest level object.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (navigation_command, communication_command)
        """
        reeval = True if self.reeval_prob == 1 else (np.random.rand() < self.reeval_prob)
        
        # Update command according to agent's movement
        obs = self.parser(obs)
        self.nav_cmd = self._update_nav_cmd(self._robot, self.robot_pos, self.nav_cmd, obs)
        self.robot_pos = self._robot.pos

        if self.nav_cmd is None or reeval:
            obj_inds = list(range(self._robot.max_obj_lvl))[::-1]  # Descending order
            
            for i in obj_inds:  # From highest level to lowest
                obs_i = obs["object"][i, :, :]
                self.nav_cmd = super()._get_closest_ind(obs_i)
                
                if self.nav_cmd is not None and self.nav_cmd != (None, None):
                    if self.comm:
                        self.comm_cmd = min(i, self.comm - 1)
                    else:
                        self.comm_cmd = 0
                    break
            else:
                # If no object is found in the surroundings
                self.comm_cmd = 0
                if self.comm:
                    # Check if there are agents with their radio on
                    # i.e., agents that see any objects
                    comm_obs = obs["comm"][0, :, :]
                    self.nav_cmd = super()._get_closest_ind(comm_obs)
        
        return self.nav_cmd, self.comm_cmd


class LeastRemainingAgentsObjPolicy(HighestLvlObjPolicy):
    """Policy that prioritizes objects with fewest remaining agents needed.
    
    Navigates to the object that requires the fewest additional agents to complete.
    """
    
    def __init__(self, robot, agent_reeval_rate=True):
        """Initialize policy.
        
        Args:
            robot: Robot instance with target intention encoding
            agent_reeval_rate: Probability of reevaluating target each step
        """
        super().__init__(robot, agent_reeval_rate=agent_reeval_rate)
        assert self._robot.share_intention == "target", "Target intention encoding required for this policy"

    def dist_grid(self, sz: int) -> np.ndarray:
        """Create a grid with distance values from center.
        
        Args:
            sz: Size of grid
            
        Returns:
            Grid with distance values
        """
        small_sz = ceil(sz / 2)
        small_grid = np.full(shape=(small_sz, small_sz), fill_value=small_sz - 1)
        
        for i in range(small_sz - 2, -1, -1):
            small_grid[:i + 1, :i + 1] = i
            
        large_grid = np.block([
            [np.flip(small_grid[1:, 1:]), np.flipud(small_grid[1:, :])],
            [np.fliplr(small_grid[:, 1:]), small_grid],
        ])
        
        return large_grid

    def _get_closest_ind(self, grid) -> Tuple[Optional[int], Optional[int]]:
        """Find closest object in grid relative to center.
        
        Args:
            grid: 2D grid of observations
            
        Returns:
            Relative coordinates (y, x) of closest object or (None, None) if grid is empty
        """
        ind = np.argwhere(grid)
        if ind.size == 0:
            # No objects found, return None
            return None, None
            
        # Find closest point
        center = grid.shape[0] // 2
        rel_ind = ind - center
        rel_abs_ind = np.sum(np.abs(rel_ind), axis=1)
        i = np.argmin(rel_abs_ind)
        
        # Returns RELATIVE coordinates of desired destination
        return tuple(rel_ind[i, :])
    
    def __call__(self, obs) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Execute policy based on observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Tuple of (navigation_command, communication_command)
        """
        DISTANCE_COEFFICIENT = 0.4  # Coefficient to weight distance in decision making
        
        reeval = True if self.reeval_prob == 1 else (np.random.rand() < self.reeval_prob)
        
        # Update command according to agent's movement
        pobs = self.parser(obs)
        self.nav_cmd = self._update_nav_cmd(self._robot, self.robot_pos, self.nav_cmd, pobs)
        self.robot_pos = self._robot.pos
        
        if self.nav_cmd is None or reeval:
            # Consider level-2+ tasks
            obj_inds = list(range(1, self._robot.max_obj_lvl))
            sum_obs = np.zeros_like(pobs["object"][0, :, :])
            
            for i in obj_inds:
                sum_obs += (i + 1) * pobs["object"][i, :, :]
                
            intention = pobs["intention"][0]
            intention[sum_obs == 0] = 0
            sum_obs -= intention
            sum_obs[sum_obs <= 0] = 0

            if not np.all(sum_obs == 0):
                # If there are level-2+ tasks with remaining agents,
                # pick the one with the least remaining agents
                sum_obs[sum_obs <= 0] = 10
                sum_obs += DISTANCE_COEFFICIENT * self.dist_grid(sum_obs.shape[0])
                argmin = np.unravel_index(np.argmin(sum_obs, axis=None), sum_obs.shape)
                center = sum_obs.shape[0] // 2
                argmin = (argmin[0] - center, argmin[1] - center)
                self.nav_cmd = argmin
                self.comm_cmd = min(2, self.comm - 1)
            else:
                # If there are no level-2+ tasks with remaining agents,
                # consider level-1 tasks
                sum_obs = pobs["object"][0, :, :]
                intention = pobs["intention"][0]
                intention[sum_obs == 0] = 0
                sum_obs -= intention
                
                if not np.all(sum_obs == 0):
                    sum_obs[sum_obs <= 0] = 10
                    sum_obs += DISTANCE_COEFFICIENT * self.dist_grid(sum_obs.shape[0])
                    argmin = np.unravel_index(np.argmin(sum_obs, axis=None), sum_obs.shape)
                    center = sum_obs.shape[0] // 2
                    argmin = (argmin[0] - center, argmin[1] - center)
                    self.nav_cmd = argmin
                    self.comm_cmd = min(1, self.comm - 1)
                else: 
                    # If no level-1 tasks are found, go for the closest highest level task
                    return super().__call__(obs)

        return self.nav_cmd, self.comm_cmd