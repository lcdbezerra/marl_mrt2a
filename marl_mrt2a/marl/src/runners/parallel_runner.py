from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import wandb

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.t_env = 0

        self.obs_shape = None
        self.step_env_info = None

        # self.train_returns = []
        # self.test_returns = []
        # self.train_stats = {}
        # self.test_stats = {}
        self.returns = {}
        self.stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

            self.obs_shape = data["obs"][0].shape[1:]

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

        self.step_env_info = None

    def _set_test_mode(self, test_mode):
        for conn in self.parent_conns:
            conn.send(("set_mode", test_mode))
            assert conn.recv(), "Setting environment mode didn't work"

    def run(self, test_mode=False):
        if not test_mode:
            return self._run()
        else:
            self._run(test_mode, log_prefix="test_")
            # self.run_perm_importance()
            return True
        
    def run_perm_importance(self):
        self.parent_conns[0].send(("get_channel_info", None))
        channel_info = self.parent_conns[0].recv()
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

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size) 
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            actions, target_updates = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                                                              test_mode=test_mode, env_info=self.step_env_info,
                                                              bs=envs_not_terminated)
            # Filter actions
            # actions = self.filter_actions_by_robot_position(actions)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1),
                "target_update": target_updates,
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "action_exec": [],
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                # "target_update": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            self.step_env_info = [0]*len(self.parent_conns)
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    # Store robot info
                    self.step_env_info[idx] = data["info"].copy()
                    data["info"].pop("robot_info")
                    action_exec = data["info"].pop("action_exec")
                    post_transition_data["action_exec"].append(action_exec)

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])                    
                    pre_transition_data["obs"].append(self._shuffle_channels(data["obs"], channels_to_shuffle))

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # Add target update rate to final_env_infos
        self._target_update_info(final_env_infos)

        # cur_stats = self.test_stats if test_mode else self.train_stats
        # cur_returns = self.test_returns if test_mode else self.train_returns
        cur_stats = self.stats[log_prefix]
        cur_returns = self.returns[log_prefix]
        # log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        
        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.returns[log_prefix]) == n_test_runs):
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

    def _target_update_info(self, info):
        assert self.batch["target_update"].shape[0] == len(info), \
            f"Unexpected behavior, batch.shape[0]={self.batch.shape[0]}, len(prev_info)={len(info)}"
        filled = self.batch["filled"].squeeze(-1)

        for i,p_info in enumerate(info):
            # Consider only filled positions in the batch
            rate = self.batch["target_update"][i, filled[i]>0]
            # Average over steps and agents
            p_info.update({
                "target_revision":  th.mean(rate[...,0], dtype=float).item(),
                "target_change":    th.mean(rate[...,1], dtype=float).item(),
            })
        return info

    def filter_actions_by_robot_position(self, actions):
        if self.step_env_info is None or self.obs_shape is None:
            return actions
        
        # Convert each robot's stored command to an action
        shape = self.obs_shape
        for worker_idx in range(len(self.parent_conns)):
            env_info = self.step_env_info[worker_idx]
            for i,e in enumerate(env_info["robot_info"]):
                pos, cmd = e
                if cmd is None or cmd==(None, None): continue
                dif = (cmd[0]-pos[0]+shape[0]//2, cmd[1]-pos[1]+shape[1]//2)
                try:
                    act = np.ravel_multi_index(dif, shape)
                    # assert dif==np.unravel_index(act, shape)
                    actions[worker_idx][i] = act
                except:
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


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_channel_info":
            remote.send(env.get_channel_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "set_mode":
            remote.send(env.set_mode(data))
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

