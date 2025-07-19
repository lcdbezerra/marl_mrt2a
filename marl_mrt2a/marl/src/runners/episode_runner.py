from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import wandb

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.step_env_info = None

        # self.train_returns = []
        # self.test_returns = []
        # self.train_stats = {}
        # self.test_stats = {}

        self.returns = {}
        self.stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.step_env_info = None

    def _set_test_mode(self, test_mode):
        self.env.set_mode(test_mode)

    def run(self, test_mode=False):
        if not test_mode:
            return self._run()
        else:
            self._run(test_mode, log_prefix="test_")
            # self.run_perm_importance()
            return True

    def run_perm_importance(self):
        channel_info = self.env.get_channel_info()
        for k,v in channel_info.items():
            if v is None: continue
            if isinstance(v, int): v = [v]
            prefix = f"permute_{k}_"
            self._run(test_mode=True, channels_to_shuffle=v, log_prefix=prefix)
            # Also log relative permutation importance
            if not self.args.wandb_sweep: continue
            # try:
            #     wandb.run.summary[f"{prefix}relative_return_mean"] = \
            #         wandb.run.summary[f"{prefix}return_mean"]/wandb.run.summary[f"test_return_mean"]
            #     wandb.run.summary[f"{prefix}relative_return_std"] = \
            #         wandb.run.summary[f"{prefix}return_std"]/wandb.run.summary[f"test_return_mean"]
            # except:
            #     pass
        return True

    def _run(self, test_mode=False, channels_to_shuffle=[], log_prefix=""):
        self._set_test_mode(test_mode)
        self.reset()
        if self.returns.get(log_prefix, None) is None:
            self.returns[log_prefix] = []
        if self.stats.get(log_prefix, None) is None:
            self.stats[log_prefix] = {}

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self._shuffle_channels(self.env.get_obs(), channels_to_shuffle)]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, target_updates = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, env_info=self.step_env_info)
            # Filter actions!

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            self.step_env_info = env_info.copy()

            action_exec = env_info["action_exec"]
            env_info.pop("robot_info", None)
            env_info.pop("action_exec", None)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "target_update": target_updates,
                "action_exec": action_exec,
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self._shuffle_channels(self.env.get_obs(), channels_to_shuffle)]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, target_updates = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions, "target_update": target_updates}, ts=self.t)

        # Add target update rate to final_env_infos
        self._target_update_info(env_info)

        # cur_stats = self.test_stats if test_mode else self.train_stats
        # cur_returns = self.test_returns if test_mode else self.train_returns
        cur_stats = self.stats[log_prefix]
        cur_returns = self.returns[log_prefix]

        # log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.returns[log_prefix]) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        if prefix not in ["", "test_"]:
            try:
                self.logger.log_stat(prefix+"relative_return_mean", np.mean(returns)/self.logger.best_return, self.t_env)
                self.logger.log_stat(prefix+"relative_return_std", np.std(returns)/self.logger.best_return, self.t_env)
            except:
                pass
        returns.clear()

        for k, v in stats.items():
            # if k.startswith("lvl"):
            #     self.logger.log_stat(prefix + k, v, self.t_env)
            # elif k != "n_episodes":
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def _set_test_mode(self, test_mode):
        self.env.set_mode(test_mode)

    def _target_update_info(self, info):
        filled = self.batch["filled"].squeeze(-1)
        # Consider only filled positions in the batch
        rate = self.batch["target_update"][0, filled[0]>0]
        # Average over steps and agents
        info.update({
            "target_revision":  th.mean(rate[...,0], dtype=float).item(),
            "target_change":    th.mean(rate[...,1], dtype=float).item(),
        })
        return info

    def filter_actions_by_robot_position(self, actions):
        if self.step_env_info is None:
            return actions
        
        # Convert each robot's stored command to an action
        shape = self.env.ind_observation_space.shape[:2]
        # print(f"SHAPE EPISODE RUNNER: {shape}")
        for i,e in enumerate(self.step_env_info["robot_info"]):
            pos, cmd = e
            if cmd is None or cmd==(None, None): continue
            dif = (cmd[0]-pos[0]+shape[0]//2, cmd[1]-pos[1]+shape[1]//2)
            try:
                act = np.ravel_multi_index(dif, shape)
                # assert dif==np.unravel_index(act, shape)
                actions[0][i] = act
            except:
                print('hey')
                pass

        return actions
    
    def _shuffle_channels(self, obs, channels_to_shuffle):
        idxs = {}
        # if isinstance(channels_to_shuffle, slice):
        #     channels_to_shuffle = [] # CONVERT SLICE TO RANGE
        if isinstance(channels_to_shuffle, slice) or len(channels_to_shuffle)>0:
            obs = np.stack(obs)
            copy_obs = obs.copy()
            for c in channels_to_shuffle:
                for _ in range(10):
                    try:
                        np.random.shuffle(obs[:,c])
                    except Exception as e:
                        print(e)
                    if not np.all(copy_obs[:,c]==obs[:,c]): break

            # Sanity check
            for c in range(obs.shape[1]):
                if c not in channels_to_shuffle:
                    assert np.all(copy_obs[:,c]==obs[:,c])
            
            return tuple([obs[i] for i in range(obs.shape[0])])
        else:
            return obs