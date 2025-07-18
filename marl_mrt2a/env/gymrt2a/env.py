"""
GyMRT2A Environment: A Multi-Robot Tasks Multi-Robot Task Allocation Environment.

This module implements a grid-based environment where multiple agents (robots) navigate
to execute tasks of different levels. The environment supports communication between
agents and various observation/action spaces.
"""

import gym
import numpy as np
import random
import wandb
import warnings
import time
from typing import Tuple, List, Optional, Union, Dict, Any

from .renderer import *
from .prettyrenderer import PrettyRenderer
from .robot import *
from .policy import *
from .controller import *
from .utils import *
from .object import *
from .map import Map
from .probability_map import *
from . import env_config

class GridWorldEnv(gym.Env):
    """
    A grid-based environment for multi-agent robot navigation and object collection.
    
    This environment simulates a grid world where multiple agents navigate to collect
    objects of different levels. It supports agent communication, different observation
    ranges, and various action spaces.
    
    Attributes:
        sz (Tuple[int, int]): Size of the grid world
        n_agents (int): Number of agents in the environment
        n_obj (Union[int, List[int]]): Number of objects per level
        view_range (int): Range of agent observation
        comm_range (int): Range of agent communication
        comm (bool): Whether agents can communicate
        max_obj_lvl (int): Maximum object level
        one_hot_obj_lvl (bool): Whether to use one-hot encoding for object levels
        obj_lvl_rwd_exp (float): Reward exponent for object levels
        Lsec (int): Number of sectors in the environment
        action_grid (bool): Whether to use grid-based actions
        share_intention (bool): Whether agents share their intentions
        respawn (bool): Whether objects respawn after collection
        discount (float): Discount factor for rewards
    """
    
    def __init__(self, 
                 sz: Tuple[int, int],
                 n_agents: int,
                 n_obj: Union[int, List[int]],
                 view_range: int = 1,
                 comm_range: int = 1,
                 comm: bool = False,
                 loop_around: bool = False,
                 render: bool = True,
                 max_obj_lvl: int = 3,
                 one_hot_obj_lvl: bool = True,
                 obj_lvl_rwd_exp: float = 2.0,
                 Lsec: int = 2,
                 action_grid: bool = True,
                 share_intention: bool = False,
                 respawn: bool = True,
                 discount: float = 0.95,
                 action_interface: str = "epymarl",
                 view_self: bool = False,
                 record: bool = False,
                 logger: List = None):
        """
        Initialize the Grid World Environment.
        
        Args:
            sz: Size of the grid world
            n_agents: Number of agents in the environment
            n_obj: Number of objects per level
            view_range: Range of agent observation
            comm_range: Range of agent communication
            comm: Whether agents can communicate
            loop_around: Whether the grid wraps around
            render: Whether to render the environment
            max_obj_lvl: Maximum object level
            one_hot_obj_lvl: Whether to use one-hot encoding for object levels
            obj_lvl_rwd_exp: Reward exponent for object levels
            Lsec: Number of sectors in the environment
            action_grid: Whether to use grid-based actions
            share_intention: Whether agents share their intentions
            respawn: Whether objects respawn after collection
            discount: Discount factor for rewards
            action_interface: Type of action interface to use
            view_self: Whether agents can see themselves
            record: Whether to record the environment
            logger: List of loggers to use
        """
        self.sz = sz
        assert sz[0] == sz[1], "Only square grids supported"
        
        # Initialize environment components
        self._map_fn = lambda: Map(sz, 
                                 max_range=max(view_range, comm_range),
                                 comm=comm,
                                 max_obj_lvl=max_obj_lvl,
                                 one_hot_obj_lvl=one_hot_obj_lvl,
                                 view_self=view_self,
                                 action_grid=action_grid,
                                 share_intention=share_intention)
                                 
        self._robot_fn = lambda pos: GymRobot(pos,
                                            view_range=view_range,
                                            comm=comm,
                                            comm_range=comm_range,
                                            max_obj_lvl=max_obj_lvl,
                                            one_hot_obj_lvl=one_hot_obj_lvl,
                                            controller=None if action_interface=="control" else PathfindingController,
                                            action_grid=action_grid,
                                            share_intention=share_intention)
                                            
        # Initialize probability maps
        self.obj_dist_fn = (lambda: SectorProbMap(sz, Lsec, respawn, max_obj_lvl)) if Lsec > 1 else lambda: ProbabilityMap(sz, respawn, max_obj_lvl)
        self.robot_dist_fn = lambda: ProbabilityMap(sz)
        
        # Store configuration
        self.loop_around = loop_around
        self.Lsec = Lsec
        self._render = render
        self.n_agents = n_agents
        self.n_obj_per_lvl = n_obj
        self.view_range = view_range
        self.comm = comm
        self.comm_range = comm_range
        self.max_obj_lvl = max_obj_lvl
        self.one_hot_obj_lvl = one_hot_obj_lvl
        self.obj_lvl_rwd_exp = obj_lvl_rwd_exp
        self.action_grid = action_grid
        self.share_intention = share_intention
        self.respawn = respawn
        self.discount = discount if respawn is False else 1
        self.parser = None

        # Validate configuration
        self._validate_config()
        
        # Convert n_obj to list of object levels
        self._convert_n_obj_to_list()
        
        # Initialize renderer if needed
        self._init_renderer(render, record)
        
        # Initialize environment
        self.reset()
        
    def _convert_n_obj_to_list(self) -> None:
        # Convert #obj per level to list of object levels
        # e.g., [3,2,1] => [1,1,1,2,2,3]
        if type(self.n_obj_per_lvl)==list:
            assert len(self.n_obj_per_lvl)<=self.max_obj_lvl, \
                f"Please provide a proper max_obj_lvl, expected max_obj_lvl>={len(self.n_obj_per_lvl)}"
            self.n_obj = []
            for i in range(len(self.n_obj_per_lvl)):
                N = self.n_obj_per_lvl[i]
                for _ in range(N):
                    self.n_obj.append(i+1)
        else:
            self.n_obj = [1]*self.n_obj

    def _validate_config(self) -> None:
        """Validate the environment configuration."""
        assert self.comm == int(self.comm), f"Invalid comm dimension provided: {self.comm}"
        assert not self.loop_around, "Loop around not implemented"
        assert isinstance(self.obj_lvl_rwd_exp, (float, int)), \
            f"Object level reward exponent should be number, got: {self.obj_lvl_rwd_exp}"

    def _init_renderer(self, render: bool, record: bool) -> None:
        """Initialize the environment renderer."""
        self.ibex = is_ibex()
        if render and not self.ibex:
            if isinstance(render, BaseRenderer):
                self.renderer = render
            else:
                self.renderer = PrettyRenderer(mode="human", 
                                            sz=self.sz, 
                                            fps=5, 
                                            obs_range=2*max(self.view_range, self.comm_range)+1,
                                            record=record)
        else:
            self.renderer = None

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Get the observation space for all agents."""
        return gym.spaces.Tuple(tuple([r.observation_space for r in self.map.robots]))

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Get the action space for all agents."""
        return gym.spaces.Tuple(tuple([r.action_space for r in self.map.robots]))

    @property
    def state_space(self) -> gym.spaces.Space:
        """Get the state space of the environment."""
        return self.map.state_space

    def reset(self) -> Tuple:
        """
        Reset the environment to its initial state.
        
        Returns:
            Tuple of initial observations for each agent
        """
        self._check_double_reset()
        if self.renderer is not None:
            self.renderer.reset()
            
        self.step_count = 0
        self.obj_pickup_count = [0] * self.max_obj_lvl
        self.map = self._map_fn()
        self.obj_dist = self.obj_dist_fn()
        self.robot_dist = self.robot_dist_fn()
        
        self._populate_robots(self.n_agents, self._robot_fn)
        self._populate_objects(self.n_obj, LevelObject)

        if len(self.map.robots) == 0:
            return tuple()

        obs = self.map.getAllObs()
        for r, o in zip(self.map.robots, obs):
            r.process_obs(o)

        self.rwd_history = []
        self.obj_count_history = []

        return obs

    def _check_double_reset(self) -> None:
        """Check if environment is being reset before first step."""
        try:
            if self.step_count == 0:
                # warnings.warn("Resetting environment before first step")
                pass
        except AttributeError:
            pass

    def step(self, actions: Tuple[Tuple[Tuple, int], ...], extra: bool = False) -> Tuple:
        """
        Execute one step in the environment.
        
        Args:
            actions: Actions for each agent in the format ((dif, comm), ...)
            extra: Whether to return extra information
            
        Returns:
            Tuple containing:
            - observations for each agent
            - rewards for each agent
            - done flags for each agent
            - info dictionary
            - extra info (if extra=True)
        """
        if extra:
            return self._step(actions)
        obs, rwd, done, info, _ = self._step(actions)
        return obs, rwd, done, info
    
    def _step(self, actions: Tuple[Tuple[Tuple, int], ...]): # ((dif, comm), ...)
        new_objs = []
        extra_info = {"collected": [], "spawned": []}
        
        # Process actions for each robot
        for i, r in enumerate(self.map.robots):
            act = actions[i]
            nav_act = r.process_action(act)
            # nav_act = r.process_action(act, next_step=False)
            # continue
            if nav_act == 0:
                continue

            next_x, next_y = getNextPosFromAction(r.pos, nav_act)
            next = self.map[next_x, next_y]
            count = 0
            
            while True:
                if isinstance(next, Robot) or next == "obstacle":
                    # Handle collision by trying alternative actions
                    nav_act = (nav_act + 1) % 9
                    if nav_act == 0:
                        nav_act = 1
                    next_x, next_y = getNextPosFromAction(r.pos, nav_act)
                    next = self.map[next_x, next_y]
                    count += 1
                    if count == 9:
                        break
                    # add penalty for collision?
                elif isinstance(next, Object):
                    if next.pickup(r):
                        # If the object can be picked up, do it
                        new_objs.append(next.lvl)
                        extra_info["collected"].append(next)
                        self.map.move(r.pos, (next_x, next_y))
                        # Reset the intention of agents that picked it up
                        next.reset_robots_cmd()
                    break
                elif not next:
                    self.map.move(r.pos, (next_x, next_y))
                    break

        # Reset objects and handle respawning
        for o in self.map.objects:
            o.reset()
            
        self._add_to_obj_count(new_objs)
        if self.respawn is True:
            new = self._populate_objects(N=new_objs)
        elif isinstance(self.respawn, (list, int, float)):
            new = self._populate_objects()
        extra_info["spawned"].extend(new)

        # Update environment state
        self.step_count += 1
        obs = self.map.getAllObs()
        # obs = tuple()
        for r, o in zip(self.map.robots, obs):
            r.process_obs(o)
            
        # Compute rewards
        rwd = sum([x**self.obj_lvl_rwd_exp for x in new_objs]) * self.discount**(self.step_count-1)
        rwd = tuple([rwd/self.n_agents] * self.n_agents)
        # rwd = tuple()
        
        # Determine if episode is done
        done = len(self.map.objects) == 0 if self.respawn is False else False
        done = [done] * self.n_agents
        
        # Prepare info dictionary
        info = {
            "robot_info": [(r.pos, r.controller.cmd) for r in self.map.robots],
            "action_exec": [True] * self.n_agents
        }
        info.update(self.env_info())

        if self._render:
            self.render()

        # Update history
        try:
            self.rwd_history.append(self.rwd_history[-1] + sum(rwd))
        except IndexError:
            self.rwd_history.append(sum(rwd))
        self.obj_count_history.append(self.obj_pickup_count.copy())

        return obs, rwd, done, info, extra_info

    def state(self) -> np.ndarray:
        """
        Get the current state of the environment.
        
        Returns:
            numpy array representing the current state
        """
        self.map.update_all()
        state = self.map.getState()
        assert self.state_space.contains(state), \
            f"State not in state_space, s: {state}, space: {self.state_space}"
        return state

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.renderer is not None:
            self.renderer.close()

    def render(self, mode: str = "human", show_viz_grid: bool = False, show_path: bool = True) -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
            show_viz_grid: Whether to show visualization grid
            show_path: Whether to show agent paths
        """
        if self.renderer is None:
            return
            
        self.renderer.blank()
        
        # Handle visualization grid
        if show_viz_grid is True:
            viz_grid = self.map.viz_grid()[:, self.map.max_range:-self.map.max_range, 
                                          self.map.max_range:-self.map.max_range]
        elif show_viz_grid:
            viz_grid = np.zeros((2, self.sz[0], self.sz[1]))
            r = self.map.robots[show_viz_grid]
            viz_grid[0, r.pos[0]-self.view_range:r.pos[0]+self.view_range+1,
                    r.pos[1]-self.view_range:r.pos[1]+self.view_range+1] = 1
            viz_grid[1, r.pos[0]-self.comm_range:r.pos[0]+self.comm_range+1,
                    r.pos[1]-self.comm_range:r.pos[1]+self.comm_range+1] = 1
        else:
            viz_grid = np.zeros((2, self.sz[0], self.sz[1]))

        if show_viz_grid:
            if self.comm:
                self.renderer.paint_grid(viz_grid[1,:,:], (230,)*3)  # comm
                self.renderer.paint_grid(viz_grid[0,:,:], (210,)*3)  # view
            else:
                self.renderer.paint_grid(viz_grid[0,:,:], (230,)*3)  # view

        # Render robots and objects
        self.renderer.render(self.map.robots, self.map.objects, 
                           action_grid=self.action_grid, 
                           show_path=show_path)

        # Render observations
        obs = tuple([self.map.getObs(r) for r in self.map.robots])
        obs = [self.parser(o) for o in obs]
        self.renderer.render_obs(obs)
        # # RENDER STATE IN MINIGRID
        # state = self.state()[:,self.map.max_range:-self.map.max_range, self.map.max_range:-self.map.max_range]
        # self.renderer.render_obs([self.parser(state)])

        # Draw grid lines
        for k in self.renderer.grid_origin.keys():
            if k == "main":
                continue
            self.renderer.draw_gridlines(grid=k)

            # Draw dashed rectangle around 1st agent observation
            # r = self.map.robots[0]
            # self.renderer.draw_dashed_rectangle((r.pos[0]-self.view_range, r.pos[1]+self.view_range), 
            #                                     (r.pos[0]+self.view_range, r.pos[1]-self.view_range), 
            #                                     grid="main")
            # self.renderer.draw_dashed_rectangle((r.pos[0]-self.comm_range, r.pos[1]+self.comm_range), 
            #                                     (r.pos[0]+self.comm_range, r.pos[1]-self.comm_range), 
            #                                     grid="main")

        self.renderer.tick()

    def screenshot(self, filename):
        if self._render:
            self.renderer.screenshot(filename)

    def layered_screenshot(self, path):
        if not self._render:
            return False
        sleep_time = 1
        
        # Render grids
        self.renderer.blank()
        for k in self.renderer.grid_origin.keys():
            self.renderer.draw_gridlines(grid=k)
        self.renderer.tick()
        time.sleep(sleep_time)
        self.renderer.screenshot(path+"_grid.png")

        # Render tasks
        for l in range(1, self.max_obj_lvl+1):
            self.renderer.blank()
            tasks = [o for o in self.map.objects if o.lvl==l]
            self.renderer.render(objects=tasks, draw_gridlines=False)
            self.renderer.tick()
            time.sleep(sleep_time)
            self.renderer.screenshot(path+f"_lvl{l}.png")
        
        # Render robots
        self.renderer.blank()
        self.renderer.render(robots=self.map.robots, draw_gridlines=False)
        self.renderer.tick()
        time.sleep(sleep_time)
        self.renderer.screenshot(path+"_robots.png")

        # Render 1st robot's range
        self.renderer.blank()
        r = self.map.robots[0]
        self.renderer.draw_dashed_rectangle((r.pos[0]-self.view_range, r.pos[1]+self.view_range), 
                                            (r.pos[0]+self.view_range, r.pos[1]-self.view_range), 
                                            grid="main")
        self.renderer.draw_dashed_rectangle((r.pos[0]-self.comm_range, r.pos[1]+self.comm_range), 
                                            (r.pos[0]+self.comm_range, r.pos[1]-self.comm_range), 
                                            grid="main")
        self.renderer.tick()
        time.sleep(sleep_time)
        self.renderer.screenshot(path+"_range.png")

        # Render all
        self.renderer.blank()
        self.renderer.render(self.map.robots, self.map.objects, action_grid=self.action_grid, show_path=True)
        self.renderer.draw_dashed_rectangle((r.pos[0]-self.view_range, r.pos[1]+self.view_range), 
                                            (r.pos[0]+self.view_range, r.pos[1]-self.view_range), 
                                            grid="main")
        self.renderer.draw_dashed_rectangle((r.pos[0]-self.comm_range, r.pos[1]+self.comm_range), 
                                            (r.pos[0]+self.comm_range, r.pos[1]-self.comm_range), 
                                            grid="main")
        self.renderer.tick()
        time.sleep(sleep_time)
        self.renderer.screenshot(path+"_all.png")


    def _populate_robots(self, N: int, robot_fn: callable, pos: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Populate the environment with robots.
        
        Args:
            N: Number of robots to add
            robot_fn: Function to create robots
            pos: Optional list of positions for robots
        """
        if N == 0:
            return
            
        if pos is None:
            avail = self.map.availability_grid()
            pos = self.robot_dist.sample(N, avail_grid=avail)
        else:
            assert len(pos) == N, f"Expected {N} positions, got {len(pos)}"

        for i in range(N):
            r = robot_fn(pos[i])
            self.map[pos[i][0], pos[i][1]] = r
            
        self.map.parser = r.parser
        self.parser = r.parser

    def populate_robots(self, N, pos=None):
        self._populate_robots(N, self._robot_fn, pos=pos)
        self.n_agents += N

    def _populate_objects(self, N: Optional[Union[int, List[int]]] = None, 
                         object_type: type = LevelObject) -> List[Object]:
        """
        Populate the environment with objects.
        
        Args:
            N: Number of objects to add or list of object levels
            object_type: Type of object to create
            
        Returns:
            List of created objects
        """
        avail = self.map.availability_grid()
        if N is None:
            N = self.obj_dist.sample_obj_num(avail_grid=avail)
            if len(N) == 0:
                return []
                
        new_objs = []

        if isinstance(N, int):
            pos = self.obj_dist.sample(N, avail_grid=avail)
            for i in range(N):
                o = object_type(pos[i])
                new_objs.append(o)
                self.map[pos[i][0], pos[i][1]] = o
        elif isinstance(N, list):
            pos = self.obj_dist.sample(len(N), avail_grid=avail)
            for i, lvl in enumerate(N):
                o = object_type(pos[i], lvl=lvl)
                new_objs.append(o)
                self.map[pos[i][0], pos[i][1]] = o
        else:
            raise TypeError(f"Unexpected argument provided to _populate_objects: {N}")
            
        return new_objs

    def _add_to_obj_count(self, new_objs: List[int]) -> None:
        """
        Update object pickup counts.
        
        Args:
            new_objs: List of object levels that were picked up
        """
        for o in new_objs:
            self.obj_pickup_count[o-1] += 1

    def obj_pickup_rate(self) -> List[float]:
        """
        Calculate the rate of object pickups per step.
        
        Returns:
            List of pickup rates for each object level
        """
        n_steps = max(1, self.step_count)
        return [n/n_steps for n in self.obj_pickup_count]

    def obj_pickup_share(self) -> List[float]:
        """
        Calculate the share of pickups for each object level.
        
        Returns:
            List of pickup shares for each object level
        """
        total = max(1, sum(self.obj_pickup_count))
        return [n/total for n in self.obj_pickup_count]

    def avg_neighbors(self) -> float:
        """
        Calculate the average number of neighbors per agent.
        
        Returns:
            Average number of neighbors
        """
        comm_range_matrix = self.map._range_matrix("comm", include_self=False)
        return np.mean(np.sum(comm_range_matrix, axis=1))

    def env_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dictionary containing environment statistics
        """
        info = dict(zip(
            [f"lvl{i+1}_obj_count" for i in range(self.max_obj_lvl)],
            self.obj_pickup_count
        ))
        
        n_steps = max(1, self.step_count)
        info.update(dict(zip(
            [f"lvl{i+1}_obj_rate" for i in range(self.max_obj_lvl)],
            self.obj_pickup_rate()
        )))
        
        info.update(dict(zip(
            [f"lvl{i+1}_obj_share" for i in range(self.max_obj_lvl)],
            self.obj_pickup_share()
        )))
        
        info.update({
            "avg_neighbors": self.avg_neighbors(),
        })
        
        return info

    def get_channel_info(self) -> Dict[str, Any]:
        """
        Get information about communication channels.
        
        Returns:
            Dictionary containing channel information
        """
        return self.map.get_channel_info()

    def set_mode(self, test_mode: bool) -> bool:
        """
        Set the environment mode.
        
        Args:
            test_mode: Whether to set test mode
            
        Returns:
            True if mode was set successfully
        """
        return True

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the environment configuration.
        
        Returns:
            Dictionary containing environment configuration
        """
        return {
            "sz": self.sz,
            "Lsec": self.Lsec,
            "n_agents": self.n_agents,
            "n_obj": self.n_obj_per_lvl,
            "max_obj_lvl": self.max_obj_lvl,
            "view_range": self.view_range,
            "comm": self.comm,
            "comm_range": self.comm_range,
            "one_hot_obj_lvl": self.one_hot_obj_lvl,
            "obj_lvl_rwd_exp": self.obj_lvl_rwd_exp,
            "action_grid": self.action_grid,
            "share_intention": self.share_intention,
            "respawn": self.respawn,
            "render": self._render,
            "discount": self.discount,
        }

##############################################################################
# META ENVIRONMENT

class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper for curriculum learning in the environment.
    
    This wrapper implements a curriculum learning approach by gradually increasing
    the difficulty of the environment based on the training progress.
    """
    
    def __init__(self, env: GridWorldEnv, setting_policy: Optional[callable] = None, **kwargs):
        """
        Initialize the curriculum wrapper.
        
        Args:
            env: The environment to wrap
            setting_policy: Optional policy for setting environment parameters
            **kwargs: Additional arguments
        """
        self.env = env
        self.setting_policy = setting_policy
        self.test_mode = False
        self.ep_counter = 0
        self.total_step_counter = 0
        self.default_config = env_config.DEFAULT_CONFIG.copy()
        self.default_config.update(env.config)
        self._load_wandb_config()

    def _load_wandb_config(self) -> None:
        """Load configuration from Weights & Biases."""
        try:
            self.config = wandb.config
        except:
            self.config = {}
        self.n_steps = self.config.get("t_max", 2_000_000)
        self.n_parallel_envs = self.config.get("batch_size_run", 1)
        self.n_steps_per_env = self.n_steps/self.n_parallel_envs

    def reset(self) -> Tuple:
        """
        Reset the environment with curriculum settings.
        
        Returns:
            Tuple of initial observations
        """
        kwargs = self.policy()
        self.ep_counter += 1
        env = GridWorldEnv(**kwargs)
        assert self._check_compatibility(self.env, env)
        self.env.close()
        self.env = env
        return self.env.reset()
    
    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step in the environment.
        
        Args:
            actions: Actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results
        """
        self.total_step_counter += 1
        return self.env.step(actions, *args)

    def policy(self) -> Dict[str, Any]:
        """
        Get the current curriculum policy settings.
        
        Returns:
            Dictionary containing environment configuration
        """
        if self.test_mode:
            return self.default_config
        elif self.total_step_counter < self.n_steps_per_env*1/6:
            config = self.default_config.copy()
            config.update(env_config.DENSE_2SECTION_LVL1)
            return config
        elif self.total_step_counter < self.n_steps_per_env*1/3:
            config = self.default_config.copy()
            config.update(env_config.SPARSE_2SECTION_LVL1)
            return config
        elif self.total_step_counter < self.n_steps_per_env*2/3:
            config = self.default_config.copy()
            config.update(env_config.SPARSE_2SECTION_LVL123)
            return config
        else:
            return self.default_config
    
    def set_mode(self, test_mode: bool) -> bool:
        """
        Set the environment mode.
        
        Args:
            test_mode: Whether to set test mode
            
        Returns:
            True if mode was set successfully
        """
        try:
            return self.env.set_mode(test_mode)
        except:
            return False
    
    def _check_compatibility(self, old: GridWorldEnv, new: GridWorldEnv) -> bool:
        """
        Check if two environments are compatible.
        
        Args:
            old: The old environment
            new: The new environment
            
        Returns:
            True if environments are compatible
        """
        assert old.observation_space == new.observation_space, \
            f"""Observation space must be kept equal
            before: {old.observation_space}, 
            after: {new.observation_space}"""
        assert old.action_space == new.action_space, \
            f"""Action space must be kept equal
            before: {old.action_space}, 
            after: {new.action_space}"""
        return True

    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about observation channels."""
        return self.env.get_channel_info()

    @property
    def state_space(self) -> gym.spaces.Space:
        """Get the state space of the environment."""
        return self.env.state_space

    def state(self) -> np.ndarray:
        """Get the current state of the environment."""
        return self.env.state()


class CurriculumEnv(gym.Env):
    """
    Environment for curriculum learning.
    
    This environment implements a curriculum learning approach by switching between
    different environment configurations during training.
    """
    
    def __init__(self, train_args: Union[Dict[str, Any], List[Dict[str, Any]]], 
                 eval_args: Union[Dict[str, Any], List[Dict[str, Any]]], 
                 render: bool = False, 
                 comm_rounds: Optional[int] = None):
        """
        Initialize the curriculum environment.
        
        Args:
            train_args: Training environment configurations
            eval_args: Evaluation environment configurations
            render: Whether to render the environment
            comm_rounds: Number of communication rounds (unused)
        """
        if not isinstance(train_args, list):
            self.train_args = [train_args]
        else:
            self.train_args = train_args

        if not isinstance(eval_args, list):
            self.eval_args = [eval_args]
        else:
            self.eval_args = eval_args

        self.render = render
        self.renderer = None
        self.env = None
        self.test_mode = False
        self.total_step_counter = 0
        self.step_count = 0
        self.ep_counter = 0

        self.reset()

    def _load_wandb_config(self) -> None:
        """Load configuration from Weights & Biases."""
        try:
            self.config = wandb.config
        except:
            self.config = {}
        self.n_steps = self.config.get("t_max", 2_000_000)
        self.n_parallel_envs = self.config.get("batch_size_run", 1)
        self.n_steps_per_env = self.n_steps/self.n_parallel_envs

    def reset(self) -> Tuple:
        """
        Reset the environment with curriculum settings.
        
        Returns:
            Tuple of initial observations
        """
        self._check_double_reset()
        kwargs = self.policy()
        if self.step_count > 0:
            # Ensure double resets are not counted
            self.ep_counter += 1
        self.step_count = 0

        env = self._init_env(kwargs)
        assert self._check_compatibility(self.env, env)
        if self.env is not None:
            self.env.close()
        self.env = env
        return self.env.reset()
    
    def _check_double_reset(self) -> None:
        """Check if environment is being reset before first step."""
        try:
            if self.step_count == 0:
                warnings.warn("Resetting environment before first step")
        except:
            pass
    
    def _init_env(self, config: Dict[str, Any]) -> GridWorldEnv:
        """
        Initialize a new environment instance.
        
        Args:
            config: Environment configuration
            
        Returns:
            New environment instance
        """
        if self.render:
            render = True if self.renderer is None else self.renderer
        else:
            render = False

        kwargs = {
            "sz": (config["size"], config["size"]),
            "n_agents": config["N_agents"],
            "n_obj": config["N_obj"],
            "render": render,
            "comm": config["N_comm"],
            "view_range": config["view_range"],
            "comm_range": config["comm_range"],
            "Lsec": config["Lsec"],
            "one_hot_obj_lvl": True,
            "obj_lvl_rwd_exp": config["obj_lvl_rwd_exp"],
            "view_self": config.get("view_self", False),
            "max_obj_lvl": 3,
            "action_grid": config["action_grid"],
            "share_intention": config["share_intention"],
            "respawn": config["respawn"],
        }
        env = GridWorldEnv(**kwargs)

        if self.render and self.renderer is None:
            self.renderer = env.renderer

        return env
    
    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step in the environment.
        
        Args:
            actions: Actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results
        """
        self.total_step_counter += 1
        self.step_count += 1
        return self.env.step(actions, *args)

    def policy(self) -> Dict[str, Any]:
        """
        Get the current curriculum policy settings.
        
        Returns:
            Dictionary containing environment configuration
        """
        if self.test_mode:
            return self.eval_args[0]
        else:
            # Choose a random environment from train_args
            return random.choice(self.train_args)
    
    def set_mode(self, test_mode: bool) -> bool:
        """
        Set the environment mode.
        
        Args:
            test_mode: Whether to set test mode
            
        Returns:
            True if mode was set successfully
        """
        self.test_mode = test_mode
        self.env.set_mode(test_mode)
        return True
    
    def _check_compatibility(self, old: Optional[GridWorldEnv], new: GridWorldEnv) -> bool:
        """
        Check if two environments are compatible.
        
        Args:
            old: The old environment
            new: The new environment
            
        Returns:
            True if environments are compatible
        """
        if old is None:
            return True
        assert old.observation_space == new.observation_space, \
            f"""Observation space must be kept equal
            before: {old.observation_space}, 
            after: {new.observation_space}"""
        assert old.action_space == new.action_space, \
            f"""Action space must be kept equal
            before: {old.action_space}, 
            after: {new.action_space}"""
        assert old.state_space == new.state_space, \
            f"""State space must be kept equal
            before: {old.state_space}, 
            after: {new.state_space}"""
        return True

    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about communication channels."""
        return self.env.get_channel_info()

    @property
    def state_space(self) -> gym.spaces.Space:
        """Get the state space of the environment."""
        return self.env.state_space

    def state(self) -> np.ndarray:
        """Get the current state of the environment."""
        return self.env.state()
    
    @property
    def action_space(self) -> gym.spaces.Space:
        """Get the action space of the environment."""
        return self.env.action_space
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        """Get the observation space of the environment."""
        return self.env.observation_space
    
    @property
    def n_agents(self) -> int:
        """Get the number of agents in the environment."""
        return self.env.n_agents


##############################################################################
# INTERFACES (deprecated)

class ActionInterface:
    """
    Base class for action interfaces.
    
    This class provides a base implementation for converting between different
    action formats.
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize the action interface.
        
        Args:
            shape: Shape of the action space
        """
        self.shape = shape

    def __call__(self, actions: Any) -> Any:
        """
        Convert actions to the desired format.
        
        Args:
            actions: Actions to convert
            
        Returns:
            Converted actions
        """
        return actions
    
    
class GridActionInterface(ActionInterface):
    """
    Interface for grid-based actions.
    
    This interface handles conversion of grid-based actions, where each action
    represents a position in a 2D grid.
    """
    
    def check_shape(self, actions: Tuple) -> bool:
        """
        Check if actions have the correct shape.
        
        Args:
            actions: Actions to check
            
        Returns:
            True if actions have correct shape
        """
        assert isinstance(actions, tuple), "Expected actions as tuple"
        for i, a in enumerate(actions):
            assert a.shape == self.shape, f"Wrong shape received, i={i}, a={a}"
            assert np.sum(a) == 1, f"Expected exactly one action to be chosen, np.sum(a)={np.sum(a)}"
        return True
    
    def __call__(self, actions: Tuple) -> List[Tuple[int, int]]:
        """
        Convert grid-based actions to movement commands.
        
        Args:
            actions: Grid-based actions
            
        Returns:
            List of movement commands
        """
        assert self.check_shape(actions)
        act = [0] * len(actions)
        for i, a in enumerate(actions):
            nav_cmd = np.unravel_index(np.argmax(a), self.shape)
            nav_cmd = (nav_cmd[0]-self.shape[0]//2, nav_cmd[1]-self.shape[1]//2)
            act[i] = nav_cmd
        return actions


class FlatActionInterface(GridActionInterface):
    """
    Interface for flattened grid-based actions.
    
    This interface handles conversion of flattened grid-based actions, where
    each action is represented as a single integer.
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize the flat action interface.
        
        Args:
            shape: Shape of the action space
        """
        self.out_shape = shape
        self.shape = (np.prod(shape),)

    def __call__(self, actions: Tuple) -> List[Tuple[int, int]]:
        """
        Convert flattened actions to movement commands.
        
        Args:
            actions: Flattened actions
            
        Returns:
            List of movement commands
        """
        act = [0] * len(actions)
        for i, a in enumerate(actions):
            nav_cmd = np.unravel_index(a, self.out_shape)
            nav_cmd = (nav_cmd[0]-self.shape[0]//2, nav_cmd[1]-self.shape[1]//2)
            act[i] = nav_cmd
        return actions
    

class LowLevelSteeringInterface(ActionInterface):
    """
    Interface for low-level steering actions.
    
    This interface handles conversion of low-level steering actions, such as
    velocity and angular velocity commands.
    """
    
    def __init__(self):
        """Initialize the low-level steering interface."""
        raise NotImplementedError("Low-level steering interface not implemented")


##############################################################################
# PARTIALLY-HARDCODED WRAPPERS

class Wrapper(gym.ActionWrapper):
    """
    Base wrapper class for the Grid World Environment.
    
    This class provides a base implementation for wrapping the Grid World Environment
    with additional functionality.
    """
    
    @property
    def state_space(self) -> gym.spaces.Space:
        """Get the state space of the wrapped environment."""
        return self.env.state_space

    def state(self) -> np.ndarray:
        """Get the current state of the wrapped environment."""
        return self.env.state()
    
    def step(self, *args) -> Tuple:
        """Execute one step in the wrapped environment."""
        return self.env.step(*args)
    
    def set_mode(self, test_mode: bool) -> bool:
        """
        Set the mode of the wrapped environment.
        
        Args:
            test_mode: Whether to set test mode
            
        Returns:
            True if mode was set successfully
        """
        try:
            return self.env.set_mode(test_mode)
        except:
            return False

    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about communication channels."""
        return self.env.get_channel_info()


class HardcodedBaseWrapper(Wrapper):
    """
    Base wrapper for hardcoded policies.
    
    This wrapper provides base functionality for implementing hardcoded policies
    in the environment.
    """
    
    def __init__(self, env: GridWorldEnv, policy: type, policy_kwargs: Dict[str, Any] = None):
        """
        Initialize the hardcoded base wrapper.
        
        Args:
            env: The environment to wrap
            policy: The policy class to use
            policy_kwargs: Additional arguments for the policy
        """
        super().__init__(env)
        self.policy_fn = policy
        self.policies = mrt2a.policy.PolicyArray(self.policy_fn, policy_kwargs=policy_kwargs or {})
        self.last_obs = None

    def reset(self) -> Tuple:
        """Reset the wrapped environment and policy."""
        obs = super().reset()
        self.policies.reset(self.env.map.robots)
        self.last_obs = obs
        return obs
    
    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Get the action space for the wrapped environment."""
        return gym.spaces.Tuple(tuple([self.ind_action_space(space) for space in self.env.action_space]))
    
    def ind_action_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """
        Get the individual action space.
        
        Args:
            space: The action space to process
            
        Returns:
            The processed action space
        """
        raise NotImplementedError
    
    def _is_dif(self, act: Any) -> bool:
        """
        Check if a navigation action is valid.
        
        Args:
            act: The action to check
            
        Returns:
            True if the action is valid
        """
        if isinstance(act, tuple):
            pass
        else:
            pass
        return True

    def _is_comm(self, comm: Any) -> bool:
        """
        Check if a communication action is valid.
        
        Args:
            comm: The communication action to check
            
        Returns:
            True if the action is valid
        """
        if isinstance(comm, tuple):
            pass
        else:
            pass
        return True
    
    def _is_cmd(self, cmd: Any) -> bool:
        """
        Check if a command is valid.
        
        Args:
            cmd: The command to check
            
        Returns:
            True if the command is valid
        """
        try:
            return self._is_dif(cmd[0]) and self._is_comm(cmd[1])
        except:
            return False


class HardcodedCommWrapper(HardcodedBaseWrapper):
    """
    Wrapper for hardcoded communication policies.
    
    This wrapper implements hardcoded communication policies while allowing
    navigation actions to be controlled externally.
    """
    
    def __init__(self, env: GridWorldEnv, 
                 policy: type = HighestObjLvlCommPolicy,
                 policy_kwargs: Dict[str, Any] = None):
        """
        Initialize the hardcoded communication wrapper.
        
        Args:
            env: The environment to wrap
            policy: The communication policy class to use
            policy_kwargs: Additional arguments for the policy
        """
        assert issubclass(policy, CommunicationPolicy)
        super().__init__(env, policy, policy_kwargs=policy_kwargs or {})

    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step with hardcoded communication.
        
        Args:
            actions: Navigation actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results
        """
        assert self.last_obs is not None, "Reset environment first"
        actions = self.filter_hardcoded(actions)
        comm = self.policies(self.last_obs)

        actions = list(zip(actions, comm))
        assert self._is_cmd(actions), "Invalid actions"

        data = self.env.step(actions, *args)
        self.last_obs = data[0]
        return data
    
    def ind_action_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """Get the individual action space for navigation."""
        return space[0]  # filter out comm space
    
    def filter_hardcoded(self, actions: Tuple) -> Tuple:
        """
        Filter out communication actions.
        
        Args:
            actions: The actions to filter
            
        Returns:
            Filtered actions
        """
        return tuple([act[0] if isinstance(act, tuple) else act for act in actions])


class HardcodedNavWrapper(HardcodedBaseWrapper):
    """
    Wrapper for hardcoded navigation policies.
    
    This wrapper implements hardcoded navigation policies while allowing
    communication actions to be controlled externally.
    """
    
    def __init__(self, env: GridWorldEnv,
                 policy: type = None,
                 policy_kwargs: Dict[str, Any] = None):
        """
        Initialize the hardcoded navigation wrapper.
        
        Args:
            env: The environment to wrap
            policy: The navigation policy class to use
            policy_kwargs: Additional arguments for the policy
        """
        assert issubclass(policy, NavigationPolicy)
        super().__init__(env, policy, policy_kwargs=policy_kwargs or {})

    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step with hardcoded navigation.
        
        Args:
            actions: Communication actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results
        """
        assert self.last_obs is not None, "Reset environment first"
        actions = self.filter_hardcoded(actions)
        nav = self.policies(self.last_obs)
        # assert self._is_comm(actions), "Communication from policy not valid"
        # assert self._is_dif(nav), "Navigation command from actor not valid"

        actions = list(zip(nav, actions))
        assert self._is_cmd(actions), "Invalid actions"

        data = self.env.step(actions, *args)
        self.last_obs = data[0]
        return data
    
    def ind_action_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """Get the individual action space for communication."""
        return space[1]  # filter out nav. space
    
    def filter_hardcoded(self, actions: Tuple) -> Tuple:
        """
        Filter out navigation actions.
        
        Args:
            actions: The actions to filter
            
        Returns:
            Filtered actions
        """
        return tuple([act[1] if isinstance(act, tuple) else act for act in actions])


class HardcodedWrapper(HardcodedBaseWrapper):
    """
    Wrapper for fully hardcoded policies.
    
    This wrapper implements both hardcoded navigation and communication policies.
    """
    
    def __init__(self, env: GridWorldEnv,
                 policy: type = HighestLvlObjPolicy,
                 **kwargs):
        """
        Initialize the hardcoded wrapper.
        
        Args:
            env: The environment to wrap
            policy: The policy class to use
            **kwargs: Additional arguments for the policy
        """
        assert issubclass(policy, JointPolicy)
        super().__init__(env, policy, policy_kwargs=kwargs)

    def step(self, actions: Optional[Tuple] = None, extra: bool = False) -> Tuple:
        """
        Execute one step with fully hardcoded policies.
        
        Args:
            actions: Optional actions (ignored)
            extra: Whether to return extra information
            
        Returns:
            Tuple containing step results
        """
        assert self.last_obs is not None, "Reset environment first"
        actions = self.policies(self.last_obs)
        assert self._is_cmd(actions), "Invalid actions"

        data = self.env.step(actions, extra)
        self.last_obs = data[0]
        return data
    
    @property
    def action_space(self) -> None:
        """Get the action space (None for fully hardcoded policies)."""
        return None


##############################################################################
# ROBUSTNESS WRAPPERS

class ErrorWrapper(Wrapper):
    """
    Base wrapper for introducing errors in the environment.
    
    This wrapper provides base functionality for introducing various types of
    errors in the environment's observations or actions.
    """
    
    def __init__(self, env: GridWorldEnv, error_rate: float = 0.1):
        """
        Initialize the error wrapper.
        
        Args:
            env: The environment to wrap
            error_rate: Probability of error occurrence
        """
        super().__init__(env)
        self.error_rate = error_rate

    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step with potential errors.
        
        Args:
            actions: Actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results with potential errors
        """
        data = self.env.step(actions, *args)
        obs = data[0]
        
        # Generate number of errors using Poisson distribution
        N_agents = len(obs)
        N_errors = [np.random.poisson(self.error_rate) for _ in range(N_agents)]

        modified_obs = self._induce_error(obs, N_errors=N_errors)
        data = (modified_obs,) + data[1:]
        return data
    
    def _induce_error(self, obs: Tuple, N_errors: List[int]) -> Tuple:
        """
        Induce errors in the observations.
        
        Args:
            obs: Original observations
            N_errors: Number of errors to induce per agent
            
        Returns:
            Modified observations with errors
        """
        raise NotImplementedError


class CommErrorWrapper(ErrorWrapper):
    """
    Wrapper for introducing communication errors.
    
    This wrapper introduces errors in the communication between agents by
    randomly removing communication channels.
    """
    
    def __init__(self, env: GridWorldEnv, error_rate: float = 0.1):
        """
        Initialize the communication error wrapper.
        
        Args:
            env: The environment to wrap
            error_rate: Probability of communication error
        """
        super().__init__(env, error_rate)

    def _induce_error(self, obs: Tuple, N_errors: List[int]) -> Tuple:
        """
        Induce errors in agent communication.
        
        Args:
            obs: Original observations
            N_errors: Number of errors to induce per agent
            
        Returns:
            Modified observations with communication errors
        """
        N_robots = len(obs)
        assert len(N_errors) == N_robots, \
            f"Expected N_errors to be of length {N_robots}, got {len(N_errors)}"
        
        # Convert tuple to list to allow item assignment
        obs = list(obs)
        
        if not self.env.share_intention:
            raise RuntimeError("Intention sharing must be enabled for communication errors")
            
        # Get intentions from each individual robot
        intention_grid = self.env.map.intention_grid
        joint_intention_grid = np.zeros_like(intention_grid)
        
        emap = self.env.map
        comm_range_matrix = emap._range_matrix("comm", include_self=False)
        
        for i, (n_err, robot) in enumerate(zip(N_errors, emap.robots)):
            in_range = list(comm_range_matrix[i])
            
            # If n_err is greater than 0, randomly select n_err agents to remove from in_range
            if n_err > 0:
                true_indices = [idx for idx, val in enumerate(in_range) if val]
                n_err = min(n_err, len(true_indices))
                if n_err > 0 and true_indices:
                    indices_to_flip = np.random.choice(true_indices, size=n_err, replace=False)
                    for idx in indices_to_flip:
                        in_range[idx] = False
            else:
                continue
                
            if len(in_range) > 0:
                joint_intention_grid[i] = np.sum(intention_grid[in_range], axis=0)
                
            mask = robot.mask
            x, y = emap.global_pos(robot.pos)
            robot_ind = emap.ind_grid[emap._channel_robots, x, y]
            assert robot_ind != -1
            
            ind_x = (x - mask.shape[1]//2, x + mask.shape[1]//2)
            ind_y = (y - mask.shape[2]//2, y + mask.shape[2]//2)
            
            grid = np.concatenate((emap.grid, joint_intention_grid[robot_ind]), axis=0)
            obs[i] = grid[:, ind_x[0]:ind_x[1]+1, ind_y[0]:ind_y[1]+1]
            obs[i] = mask * obs[i]
            
            if not emap.view_self:
                obs[i][emap._channel_robots, mask.shape[1]//2, mask.shape[2]//2] = 0
                
            assert obs[i].shape == robot.observation_space.shape, "Invalid observation shape"
            
        return tuple(obs)


class ObsErrorWrapper(ErrorWrapper):
    """
    Wrapper for introducing observation errors.
    
    This wrapper introduces errors in agent observations by randomly flipping
    pixels in the observation grid.
    """
    
    def __init__(self, env: GridWorldEnv, error_rate: float = 0.1):
        """
        Initialize the observation error wrapper.
        
        Args:
            env: The environment to wrap
            error_rate: Probability of observation error
        """
        super().__init__(env, error_rate)

    def _induce_error(self, obs: Tuple, N_errors: List[int]) -> Tuple:
        """
        Induce errors in agent observations.
        
        Args:
            obs: Original observations
            N_errors: Number of errors to induce per agent
            
        Returns:
            Modified observations with errors
        """
        N_robots = len(obs)
        assert len(N_errors) == N_robots, \
            f"Expected N_errors to be of length {N_robots}, got {len(N_errors)}"
        
        emap = self.env.map
        for i, (n_err, robot) in enumerate(zip(N_errors, emap.robots)):
            if n_err > 0:
                # Choose n_err pixels in the observation
                channels = [emap._channel_robots] + emap._channel_objects
                view_range = robot.view_range
                sz = robot.mask.shape[1:]
                valid_obs = obs[i][channels,
                                 sz[0]//2-view_range:sz[0]//2+view_range,
                                 sz[1]//2-view_range:sz[1]//2+view_range]
                shape = valid_obs.shape
                
                # Randomly select n_err pixels to flip
                indices = np.random.choice(np.prod(shape), size=n_err, replace=False)
                valid_obs.flat[indices] = 1 - valid_obs.flat[indices]
                
                obs[i][channels,
                      sz[0]//2-view_range:sz[0]//2+view_range,
                      sz[1]//2-view_range:sz[1]//2+view_range] = valid_obs
                      
        return obs


class EPyMARLWrapper(Wrapper):
    """
    Wrapper for EPyMARL compatibility.
    
    This wrapper adapts the environment to be compatible with the EPyMARL
    framework by flattening the action space.
    """
    
    def __init__(self, env: GridWorldEnv, **kwargs):
        """
        Initialize the EPyMARL wrapper.
        
        Args:
            env: The environment to wrap
            **kwargs: Additional arguments
        """
        if isinstance(env, HardcodedWrapper):
            raise RuntimeError("EPyMARL wrapping a hardcoded environment")
            
        assert isinstance(env, (GridWorldEnv, CurriculumWrapper, CurriculumEnv,
                              HardcodedCommWrapper, HardcodedNavWrapper,
                              ObsErrorWrapper, CommErrorWrapper)), \
            f"Invalid environment: {env}"
            
        super().__init__(env, **kwargs)
        self._compute_action_shape()

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Get the flattened action space."""
        return gym.spaces.Tuple(tuple([self.ind_action_space(space) 
                                     for space in self.env.action_space]))
    
    def _compute_action_shape(self) -> None:
        """Compute the shape of the flattened action space."""
        space = self.env.action_space[0]  # Assume homogeneous agents

        # Navigation space
        if isinstance(self.env, (GridWorldEnv, CurriculumWrapper, CurriculumEnv, ErrorWrapper)):
            nav_space = space[0]
        elif isinstance(self.env, HardcodedCommWrapper):
            nav_space = space
        else:
            nav_space = None

        if nav_space is None:
            self.nav_shape = tuple()
        elif isinstance(nav_space, gym.spaces.Tuple):
            self.nav_shape = (nav_space[0].n, nav_space[1].n)
        elif isinstance(nav_space, gym.spaces.Discrete):
            self.nav_shape = (nav_space.n,)
        else:
            raise RuntimeError(f"Unexpected behavior, navigation space: {nav_space}")
        
        # Communication space
        if isinstance(self.env, (GridWorldEnv, CurriculumWrapper, CurriculumEnv, ErrorWrapper)):
            comm_space = space[1]
        elif isinstance(self.env, HardcodedNavWrapper):
            comm_space = space
        else:
            comm_space = None
            
        self.comm_shape = (comm_space.n,) if comm_space is not None else (1,)
        self.shape = self.nav_shape + self.comm_shape
    
    def ind_action_space(self, space: gym.spaces.Space) -> gym.spaces.Discrete:
        """
        Get the individual action space.
        
        Args:
            space: The action space to process
            
        Returns:
            Flattened discrete action space
        """
        n_dim = np.prod(self.shape)
        return gym.spaces.Discrete(n_dim)

    def step(self, actions: Tuple) -> Tuple:
        """
        Execute one step with flattened actions.
        
        Args:
            actions: Flattened actions for each agent
            
        Returns:
            Tuple containing step results
        """
        actions = self.flat_to_shape(actions)
        return self.env.step(actions)
    
    def flat_to_shape(self, actions: Tuple) -> List[Tuple]:
        """
        Convert flattened actions to structured format.
        
        Args:
            actions: Flattened actions
            
        Returns:
            Structured actions
        """
        action_lst = [0] * len(actions)
        for i, a in enumerate(actions):
            action = np.unravel_index(a, self.shape)
            if len(self.nav_shape) > 0:
                *nav_cmd, comm = action
                assert len(nav_cmd) in [1, 2], "Unexpected behavior"
                if len(nav_cmd) > 1:
                    nav_cmd = (nav_cmd[0]-self.shape[0]//2, nav_cmd[1]-self.shape[1]//2)
            else:
                nav_cmd = 0
                comm = action

            action_lst[i] = (nav_cmd, comm)
        return action_lst


class ObservationVisualization(Wrapper):
    """
    Wrapper for visualizing agent observations.
    
    This wrapper provides interactive visualization of agent observations
    during environment execution.
    """
    
    def __init__(self, env: GridWorldEnv):
        """
        Initialize the observation visualization wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        sz = self.env.observation_space[0].shape[1:]
        self.renderer = ObservationRenderer(sz)
        self.parser = self.env.parser
        self.robot_id = 0
        self.key = "obstacle"
        self.channel = 0
        self.last_obs = None

    def reset(self) -> Tuple:
        """Reset the environment and visualization."""
        self.last_obs = None
        return self.env.reset()

    def step(self, actions: Tuple, *args) -> Tuple:
        """
        Execute one step with observation visualization.
        
        Args:
            actions: Actions for each agent
            *args: Additional arguments
            
        Returns:
            Tuple containing step results
        """
        while True:
            if self.last_obs is None:
                break
                
            try:
                prompt = input("Viz:\t")
                if prompt == "":
                    break
                    
                prompt = prompt.split(',')
                assert len(prompt) <= 2, f"Invalid visualization prompt: {','.join(prompt)}"
                
                self.key = prompt[0]
                self.channel = 0 if len(prompt) == 1 else int(prompt[1])
                
                im = self.parser(self.last_obs[self.robot_id])[self.key][self.channel]
                self.renderer.imshow(im)
            except Exception as e:
                print("try again...")
                print(e)
                continue

        data = self.env.step(actions, *args)
        self.last_obs = data[0]
        im = self.parser(self.last_obs[self.robot_id])[self.key][self.channel]
        self.renderer.imshow(im)

        return data


class ManualControl(Wrapper):
    """
    Wrapper for manual control of agents.
    
    This wrapper allows manual control of agents through user input.
    """
    
    def reset(self) -> Tuple:
        """Reset the environment and control state."""
        self.last_info = None
        return super().reset()

    def step(self, *args, actions: Optional[Tuple] = None) -> Tuple:
        """
        Execute one step with manual control.
        
        Args:
            *args: Additional arguments
            actions: Optional actions (ignored)
            
        Returns:
            Tuple containing step results
        """
        actions = self._get_actions_from_input()
        data = self.env.step(actions, *args)
        self.last_info = data[3]
        return data

    def _get_actions_from_input(self) -> List[Tuple]:
        """
        Get actions from user input.
        
        Returns:
            List of actions for each agent
        """
        assert isinstance(self.env.observation_space, gym.spaces.Tuple)
        n_agents = len(self.env.observation_space)
        actions = [None] * n_agents
        
        print("Please enter the command for each robot")
        for i in range(n_agents):
            act = False
            while not act:
                act = input(f"Robot {i}:\t")
                act = self._process_act(act)
                if act is None:
                    break
                elif act is False:
                    print("try again...")
            actions[i] = act

        actions = self._add_current_target(actions)
        return actions

    def _process_act(self, act: str) -> Optional[Tuple]:
        """
        Process a user input action.
        
        Args:
            act: User input action string
            
        Returns:
            Processed action or None/False if invalid
        """
        if len(act) == 0:
            return None
            
        try:
            act = act.split(',')
            assert len(act) == 3
            act = [int(a) for a in act]
            return tuple(act)
        except (ValueError, AssertionError):
            return False

    def _add_current_target(self, actions: List[Optional[Tuple]]) -> List[Tuple]:
        """
        Add current target to actions.
        
        Args:
            actions: List of actions to process
            
        Returns:
            List of actions with current target
        """
        for i, act in enumerate(actions):
            if act is None:
                actions[i] = ((0, 0), 0)
        return actions


if __name__=="__main__":
    config = {
        "buffer_size": 10,
        # "config": "vdn",
        "config": "mappo", 
        "critic_type": "cnn_cv_critic",
        "env_config": "gridworld", "agent": "cnn",
        "agent_arch": "resnet;conv2d,64,1;relu;interpolate,2;conv2d,1,1;relu;interpolate,1.7&",
        "critic_arch": "resnet&batchNorm1d;linear,128;relu;linear,32;relu",
        "strategy": "cnn",
        # "strategy": "hardcoded",
        # "env_config": "gymma",
        "hidden_dim": 512,
        "obs_agent_id": False,
        "gymrt2a.Lsec": 2,
        "gymrt2a.N_agents": 10,
        "gymrt2a.N_comm": 2,
        # "gymrt2a.N_comm": 0,
        "gymrt2a.N_obj": [4, 3, 3],
        "gymrt2a.comm_range": 8,
        # "gymrt2a.size": 40,
        "gymrt2a.size": 20,
        "gymrt2a.view_range": 8,
        "gymrt2a.action_grid": True,
        "action_grid": True,
        "gymrt2a.share_intention": "path",
        "share_intention": "path",
    }
    kwargs={
            "sz": (config["gymrt2a.size"], config["gymrt2a.size"]),
            "n_agents": config["gymrt2a.N_agents"],
            "n_obj": config["gymrt2a.N_obj"],
            "render": True,
            "comm": config["gymrt2a.N_comm"],
            # "hardcoded_comm": config["hardcoded_comm"],
            "view_range": config["gymrt2a.view_range"],
            "comm_range": config["gymrt2a.comm_range"],
            "Lsec": config["gymrt2a.Lsec"],
            "one_hot_obj_lvl": True,
            "max_obj_lvl": 3,
            "action_grid": config["gymrt2a.action_grid"],
            "share_intention": config["gymrt2a.share_intention"],
        }
    env = GridWorldEnv(**kwargs)
    env = HardcodedCommWrapper(env)

    print(env)