from smac.env import MultiAgentEnv
import gym
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from gymrt2a.env import *

class TimeLimit(GymTimeLimit):
    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def set_mode(self, test_mode):
        try:
            return self.env.set_mode(test_mode)
        except:
            return False
    

class GridworldWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.comm_rounds = kwargs.pop("comm_rounds", 0)
        self.episode_limit = time_limit*(self.comm_rounds+1)
        self.hardcoded = kwargs.pop("hardcoded", False)
        self.curriculum = kwargs.pop("curriculum", False)
        self.comm_error = kwargs.pop("comm_error", False)
        self.obs_error = kwargs.pop("obs_error", False)
        assert self.hardcoded in [False, "comm", "nav"], f"Unexpected value for env_args.hardcoded: {self.hardcoded}"
        # env = gym.make(f"{key}", comm_rounds=self.comm_rounds)
        env = gym.make(f"{key}")

        # if self.curriculum:
        #     env = CurriculumWrapper(env)
        
        if self.hardcoded == "comm":
            env = HardcodedCommWrapper(env, policy=TargetObjLvlCommPolicy)
        elif self.hardcoded == "nav":
            env = HardcodedNavWrapper(env)
            
        if self.comm_error:
            env = CommErrorWrapper(env, error_rate=self.comm_error)
        if self.obs_error:
            env = ObsErrorWrapper(env, error_rate=self.obs_error)

        env = EPyMARLWrapper(env)
        self._env = TimeLimit(env, max_episode_steps=self.episode_limit)
        # self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            raise NotImplementedError("Pretrained wrapper with Gridworld")

        self.n_agents = self._env.n_agents
        self._obs = None

        self.ind_action_space = self._env.action_space[0] # assume homogeneous agents
        self.ind_observation_space = self._env.observation_space[0] # assume homogeneous agents

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [self._channels_first(x) for x in self._obs]
        if isinstance(done, bool): done = [done]
        return float(sum(reward)), all(done), info
    
    def set_mode(self, test_mode):
        try:
            return self._env.set_mode(test_mode)
        except:
            return False


    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        # raise self._obs[agent_id] # raise?
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._channels_first(self.ind_observation_space.shape)

    def get_state(self):
        return self._channels_first(self._env.state())

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self._channels_first(self._env.state_space.shape)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """

        # For now, assume all actions are always valid
        return self.ind_action_space.n * [1]

        # valid = flatdim(self._env.action_space[agent_id]) * [1]
        # invalid = [0] * (self.longest_action_space.n - len(valid))
        # return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.ind_action_space.n

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [self._channels_first(x) for x in self._obs]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def _channels_first(self, obs):
        if isinstance(obs, np.ndarray):
            assert len(obs.shape)==3, "Expected 3D tensor: WxHxC"
            assert obs.shape[1]==obs.shape[2], f"All library should be in channel-first standard, but got shape: {obs.shape}"
            # raise NotImplementedError("Channels-first")
            # return np.transpose(obs, [2,0,1])
        elif isinstance(obs, tuple):
            assert len(obs)==3, "Expected 3D shape: WxHxC"
            assert obs[1]==obs[2]
            # raise NotImplementedError("Channels-first got tuple")
            # return (obs[2], obs[0], obs[1])
        else:
            raise TypeError("Unexpected type")
        return obs
        
    def get_channel_info(self):
        return self._env.get_channel_info()