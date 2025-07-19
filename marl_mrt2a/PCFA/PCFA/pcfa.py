# Copyright 2025 Lucas Bezerra
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gymrt2a
from gymrt2a.env import GridWorldEnv, Wrapper
from gymrt2a.robot import GymRobot
from gymrt2a.object import LevelObject
import gymrt2a.astar as AStar
from copy import deepcopy

class Agent:
    def __init__(self, robot: GymRobot, id: int, lmb=1, pathfinder=None):
        self.r = robot
        self.id = id
        self.view_range = robot.view_range
        self.comm_range = robot.comm_range
        self.lmb = lmb
        self.pathfinder = pathfinder
        # self.comm_manager = comm_manager
        self.reset()

    def assign_comm_manager(self, comm_manager):
        self.comm_manager = comm_manager
    
    def distance_to(self, task, discard_obstacles=[]):
        discard_obstacles = [o.pos for o in discard_obstacles]
        if self.pathfinder is None:
            return max(abs(self.pos[0]-task.pos[0]), abs(self.pos[1]-task.pos[1]))
        else:
            return self.pathfinder.distance_between(self.id, self.pos, task.pos, discard_obstacles)
    
    def distance_between(self, p1, p2, discard_obstacles=[]):
        discard_obstacles = [o.pos for o in discard_obstacles]
        if self.pathfinder is None:
            return max(abs(p1.pos[0]-p2.pos[0]), abs(p1.pos[1]-p2.pos[1]))
        else:
            return self.pathfinder.distance_between(self.id, p1.pos, p2.pos, discard_obstacles)
        # return max(abs(p1.pos[0]-p2.pos[0]), abs(p1.pos[1]-p2.pos[1]))

    def within_range(self, task, type="view"):
        range = self.view_range if type=="view" else self.comm_range
        return self.distance_to(task) <= range

    def reset(self):
        self.path_list = []
        self.time_table = []
        self.pos = self.r.pos
        self.no_task = False

    def receive_message(self, message, type):
        pass

    def advertisement_preparation(self, tasks):
        self.app_letter = []
        self.off_letter = []
        self.winning_ad = {}
        self.fitness = [0]*len(tasks)
        zmax = self.comm_manager.get_num_neighbors(self)+1
        t_available = 0 if len(self.time_table)==0 else self.time_table[-1]
        pos_available = self if len(self.path_list)==0 else tasks[self.path_list[-1]]
        discard_obstacles = [tasks[path] for path in self.path_list[:-1]]

        for i,t in enumerate(tasks):
            if (zmax >= t.lvl) and self.within_range(t, type="view") and not t.assigned:
            # if (zmax >= t.lvl) and self.within_range(t, type="view"):
                weight = t.lvl**2
                self.fitness[i] = weight*np.exp(-self.lmb*(t_available + \
                                                           self.distance_between(pos_available, t, 
                                                                                 discard_obstacles)))
        
        if sum(self.fitness) == 0:
            self.no_task = True
            best_task = -1
            self.winning_ad[best_task] = (self.id, 0)
        else:        
            best_task = np.argmax(self.fitness)
            self.winning_ad[best_task] = (self.id, self.fitness[best_task])

        self.target_task = best_task
        self.last_ad = deepcopy(self.winning_ad)

    def broadcast_ad(self):
        self.comm_manager.broadcast_ad(self, self.last_ad)

    def receive_ad(self, message):
        for task_id, (agent_id, fitness) in message.items():
            curr_agent_id, curr_fitness = self.winning_ad.get(task_id, (0, 0))
            if (fitness > curr_fitness) or \
               (fitness == curr_fitness and agent_id < curr_agent_id):
                self.winning_ad[task_id] = (agent_id, fitness)
        self.last_ad = deepcopy(self.winning_ad)

    def application(self, tasks):
        if self.no_task:
            return
        t_available = 0 if len(self.time_table)==0 else self.time_table[-1]
        pos_available = self if len(self.path_list)==0 else tasks[self.path_list[-1]]
        discard_obstacles = [tasks[path] for path in self.path_list[:-1]]
        # resume = t_available + self.distance_to(tasks[self.target_task])
        resume = t_available + self.distance_between(pos_available, tasks[self.target_task],
                                                     discard_obstacles)
        app = (self.id, self.target_task, resume)
        if self.winning_ad[self.target_task][0] == self.id:
            self.receive_app(app)
        else:
            # If not PM, apply to the PM of the target task
            self.send_app(app, to=self.winning_ad[self.target_task][0])

    def send_app(self, app, to):
        self.comm_manager.send_app(self, app, to)

    def receive_app(self, app):
        self.app_letter.append(app)

    def send_off(self, off, to):
        self.comm_manager.send_off(self, off, to)

    def receive_off(self, off):
        self.off_letter.append(off)

    def team_building(self, tasks):
        # Check if one was assigned PM
        if not self.no_task:
            for task_id, (pm_id, _) in self.winning_ad.items():
                if pm_id == self.id:
                    task_lvl = tasks[task_id].lvl
                    if len(self.app_letter) >= task_lvl:
                        self.app_letter = sorted(self.app_letter, key=lambda x: x[2], reverse=False)
                        apps = self.app_letter[:task_lvl]
                        arrival_time = max([app[2] for app in apps])
                        off = (self.id, task_id, arrival_time)
                        for app in apps:
                            self.send_off(off, app[0])
                        tasks[task_id].assigned = True

        # Check received offers
        if len(self.off_letter) > 0:
            assert len(self.off_letter) == 1
            self.path_list.append(self.off_letter[0][1])
            self.time_table.append(self.off_letter[0][2])

        self.app_letter = []
        self.off_letter = []


class Task:
    def __init__(self, t: LevelObject):
        self.task = t
        self.lvl = t.lvl
        self.pos = t.pos
        self.assigned = False

    def reset(self):
        self.assigned = False


class CommManager:
    def __init__(self, agents):
        self.agents = agents

    def update_comm_mat(self, comm_mat):
        self.comm_mat = comm_mat

    def broadcast_ad(self, sender, message):
        sender_id = self.agents.index(sender)
        for i, agent in enumerate(self.agents):
            if self.comm_mat[sender_id][i]:
                agent.receive_ad(message)

    def broadcast(self, sender, message):
        sender_id = self.agents.index(sender)
        for i, agent in enumerate(self.agents):
            if self.comm_mat[sender_id][i]:
                agent.receive_message(message)
    
    def get_num_neighbors(self, agent):
        agent_id = self.agents.index(agent)
        return np.sum(self.comm_mat[agent_id])-1 # not counting itself
    
    def send_app(self, sender, message, to_id):
        sender_id = self.agents.index(sender)
        to = self.agents[to_id]
        if self.comm_mat[sender_id][to_id]:
            to.receive_app(message)

    def send_off(self, sender, message, to_id):
        sender_id = self.agents.index(sender)
        to = self.agents[to_id]
        if self.comm_mat[sender_id][to_id]:
            to.receive_off(message)


class Pathfinder:
    def __init__(self, env):
        self.env = env
        self.obstacle_grid = [None]*env.n_agents
        self.lvl1_obj_grid = [None]*env.n_agents

    def update_obs(self, obs):
        for i, o in enumerate(obs):
            self.obstacle_grid[i] = self.env.map.robots[i].parser.not_passable_grid(o)
            self.lvl1_obj_grid[i] = self.env.map.robots[i].parser(o)["object"][0]
            assert len(self.lvl1_obj_grid[i].shape) == 2

    def convert_to_obs_coords(self, i, p):
        return (p[0]-self.env.map.robots[i].pos[0], p[1]-self.env.map.robots[i].pos[1])

    def find_path(self, i, p1, p2, discard_obstacles=[]):
        p1, p2 = self.convert_to_obs_coords(i, p1), self.convert_to_obs_coords(i, p2)
        discard_obstacles = [self.convert_to_obs_coords(i, o) for o in discard_obstacles]

        obstacle_grid = self.obstacle_grid[i].copy()
        lvl1_obj_grid = self.lvl1_obj_grid[i].copy()
        # (0, 0) and dif shouldn't be obstacles
        obstacle_grid[p1[0] + obstacle_grid.shape[0]//2, p1[1] + obstacle_grid.shape[1]//2] = 0
        obstacle_grid[p2[0] + obstacle_grid.shape[0]//2, p2[1] + obstacle_grid.shape[1]//2] = 0
        for o in discard_obstacles:
            obstacle_grid[o[0] + obstacle_grid.shape[0]//2, o[1] + obstacle_grid.shape[1]//2] = 0
        # try:
        #     obstacle_grid[dif[0] + obstacle_grid.shape[0]//2, dif[1] + obstacle_grid.shape[1]//2] = 0
        # except:
        #     pass
        grid = AStar.ObsGrid(obstacle_grid, lvl1_obj_grid)
        came_from, cost_so_far = AStar.a_star_search(grid, start=p1, goal=p2)
        path = AStar.reconstruct_path(came_from, start=p1, goal=p2)
        if len(path) == 0: return None
        # convert to global coordinates if needed

        return path[1:] # without starting position

    def distance_between(self, i, p1, p2, discard_obstacles=[]):
        path = self.find_path(i, p1, p2, discard_obstacles=discard_obstacles)
        if path is None:
            return 1e3
        else:
            return len(path)


class PCFA(Wrapper):
    """
    Project Manager-oriented Coalition Formation Algorithm (PCFA) wrapper for GyMRT²A environments.
    
    This class implements the market-based task allocation algorithm described in:
    
        Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment 
        for Cooperative Timing Missions in Dynamic Environments. Journal of Intelligent 
        and Robotic Systems, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x
    
    The algorithm provides a decentralized approach for multi-agent task allocation in 
    dynamic environments with limited communication range. It extends baseline algorithms 
    to handle time-varying network topology including isolated subnetworks.
    
    The algorithm follows a four-phase market-based approach:
    1. Advertisement Preparation: Agents calculate fitness for available tasks
    2. Consensus on Project Manager: Agents reach consensus on task management
    3. Application: Agents apply to project managers for task assignment  
    4. Team Building: Project managers form teams and assign tasks
    
    Args:
        env (GridWorldEnv): The base GyMRT²A environment to wrap
        config (dict): Configuration dictionary with the following optional keys:
            - market_loop_length: Number of consensus loops (default: True for automatic)
            - lambda: Time discount factor for fitness calculation (default: 1)
            - pathfinder: Enable A* pathfinding (default: False)
    """
    def __init__(self, env: GridWorldEnv, config: dict):
        super().__init__(env)
        self.env = env
        self.config = config
        self.consensus_loops = config.get("market_loop_length", True)
        self.lmb = config.get("lambda", 1)
        self.enable_pathfinding = config.get("pathfinder", False)

    def reset(self):
        self.t = 0
        self.consensus_steps = []
        self.obs = self.env.reset()
        self.pathfinder = Pathfinder(self.env)
        self.pathfinder.update_obs(self.obs)
        self.agents = [Agent(r,i, lmb=self.lmb, pathfinder=(self.pathfinder if self.enable_pathfinding else None)) \
                       for i,r in enumerate(self.env.map.robots)]
        self.comm_manager = CommManager(self.agents)
        self.update_comm_mat()
        for agent in self.agents:
            agent.assign_comm_manager(self.comm_manager)
        self.tasks = self.env.map.objects
        return self.obs
    
    def update_comm_mat(self):
        if self.env.comm_range is True or self.env.comm_range>=max(self.env.sz):
            n_agents = len(self.agents)
            self.comm_manager.update_comm_mat(np.ones((n_agents, n_agents), dtype=bool))
        else:
            self.comm_manager.update_comm_mat(self.env.map._range_matrix("comm", include_self=True))

    def step(self, actions=None):
        # Initialize 
        for agent in self.agents:
            agent.reset()
        self.update_comm_mat()
        self.pathfinder.update_obs(self.obs)
        tasks = [Task(t) for t in self.tasks if t.remaining_robots()>0 and any([a.within_range(t) for a in self.agents])]

        for consensus_t in range(100):
            # Advertisement Preparation
            for agent in self.agents:
                agent.advertisement_preparation(tasks)

            # Consensus on PM
            loops = len(self.agents) if self.consensus_loops is True else min(len(self.agents), self.consensus_loops)
            for _ in range(loops):
                for agent in self.agents:
                    agent.broadcast_ad()

            # Sanity check: ensure all agents have reached consensus (only works if comm graph is fully connected)
            # assert all([a1.winning_ad == a2.winning_ad for a1,a2 in zip(self.agents[:-1], self.agents[1:])])

            # Application
            for agent in self.agents:
                agent.application(tasks)

            # Team Building
            for agent in self.agents:
                agent.team_building(tasks)
            if all([t.assigned for t in tasks]):
                break
        self.consensus_steps.append(consensus_t+1)
        
        # Interact with environment
        # Update tasks
        actions = []
        for agent in self.agents:
            if len(agent.path_list)==0:
                dif = (np.random.randint(low=-self.env.view_range, high=self.env.view_range+1),
                       np.random.randint(low=-self.env.view_range, high=self.env.view_range+1))
                act = (dif, 0)
                # act = ((0,0), 0)
                actions.append(act)
            else:
                target = tasks[agent.path_list[0]].pos
                dif = (target[0]-agent.pos[0], target[1]-agent.pos[1])
                act = (dif, 0)
                actions.append(act)
        
        self.obs, rwd, done, info, tasks_info = self.env.step(actions, extra = True)
        # for t in tasks_info["collected"]:
        #     self.tasks.remove(t)
        # for t in tasks_info["spawned"]:
        #     self.tasks.append(t)
        self.t += 1

        return self.obs, rwd, done, info 