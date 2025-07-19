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

"""
PCFA - Project Manager-oriented Coalition Formation Algorithm

A Python package implementing the Project Manager-oriented Coalition Formation
Algorithm (PCFA) for multi-agent task allocation in robotic systems.

This implementation is based on the market-based task allocation algorithm described in:

    Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment 
    for Cooperative Timing Missions in Dynamic Environments. Journal of Intelligent 
    and Robotic Systems, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x

The algorithm provides a decentralized approach for task assignment of multiple 
unmanned vehicles in dynamic environments with limited communication range, 
particularly for cooperative timing missions that cannot be performed by a single vehicle.
"""

from .pcfa import PCFA, Agent, Task, CommManager, Pathfinder

# Default configuration for PCFA
DEFAULT_CONFIG = {
    "buffer_size": 10,
    "config": "mappo", 
    "env_config": "gridworld", 
    "agent": "cnn",
    "agent_arch": "unet,8,1,2&",
    "critic_arch": "unet,8,1,2&batchNorm1d;linear,50;relu",
    "strategy": "cnn",
    "hidden_dim": 512,
    "obs_agent_id": False,
    "gymrt2a.Lsec": 2,
    "gymrt2a.N_agents": 10,
    "env_args.hardcoded": "comm",
    "agent_distance_exp": 1.,
    "gymrt2a.N_comm": 4,
    "gymrt2a.N_obj": [4, 3, 3], 
    "gymrt2a.comm_range": 20,
    "gymrt2a.size": 20,
    "gymrt2a.view_range": 4,
    "gymrt2a.action_grid": True,
    "gymrt2a.respawn": True,
    "gymrt2a.obj_lvl_rwd_exp": 2.,
    "action_grid": True,
    "current_target_factor": 2.,
    "agent_reeval_rate": None,
    "filter_avail_by_objects": True,
    "gymrt2a.share_intention": "channel",
    "share_intention": "channel",
    "seed": 10,
    "t_max": 2_000_000,
    "env_args.curriculum": True,
}

# Default environment kwargs for GridWorldEnv
DEFAULT_ENV_KWARGS = {
    "sz": (DEFAULT_CONFIG["gymrt2a.size"], DEFAULT_CONFIG["gymrt2a.size"]),
    "n_agents": DEFAULT_CONFIG["gymrt2a.N_agents"],
    "n_obj": DEFAULT_CONFIG["gymrt2a.N_obj"],
    "comm": DEFAULT_CONFIG["gymrt2a.N_comm"],
    "view_range": DEFAULT_CONFIG["gymrt2a.view_range"],
    "comm_range": DEFAULT_CONFIG["gymrt2a.comm_range"],
    "Lsec": DEFAULT_CONFIG["gymrt2a.Lsec"],
    "one_hot_obj_lvl": True,
    "obj_lvl_rwd_exp": DEFAULT_CONFIG["gymrt2a.obj_lvl_rwd_exp"],
    "max_obj_lvl": 3,
    "action_grid": DEFAULT_CONFIG["gymrt2a.action_grid"],
    "share_intention": DEFAULT_CONFIG["gymrt2a.share_intention"],
    "respawn": DEFAULT_CONFIG["gymrt2a.respawn"],
    "render": False,
    "view_self": True,
}

__version__ = "0.1.0"
__author__ = "PCFA Team"
__all__ = ["PCFA", "Agent", "Task", "CommManager", "Pathfinder", "DEFAULT_CONFIG", "DEFAULT_ENV_KWARGS"]