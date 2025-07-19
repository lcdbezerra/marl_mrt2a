import os
from time import time
import wandb
import numpy as np
import argparse
from multiprocessing import cpu_count
from copy import deepcopy
from tqdm import trange

import gym
from gym.envs.registration import register
from PCFA.pcfa import PCFA
import gymrt2a

usr_name = os.environ["USER"]
IBEX = True if os.path.exists("/ibex/") else False
online = None
base_dir = f"/home/{usr_name}/code/PCFA/"

parser = argparse.ArgumentParser(description="Run PCFA")

DEFAULT_CONFIG = {
    "gymrt2a.Lsec": 2,
    "gymrt2a.N_agents": 10,
    "env_args.hardcoded": "comm",
    "agent_distance_exp": 1.,
    "gymrt2a.N_comm": 4,
    "gymrt2a.N_obj": [4, 3, 3],
    "gymrt2a.comm_range": 8,
    "gymrt2a.size": 20,
    "gymrt2a.view_range": 8,
    "gymrt2a.action_grid": True,
    "gymrt2a.respawn": .05,
    "gymrt2a.obj_lvl_rwd_exp": 2.,
    "gymrt2a.share_intention": "channel",
    "seed": 72,
    "t_max": 250_000,
}

def run_PCFA(env):
    rwd_lst = []
    consensus_lst = []
    obj_pickup_rate = []
    try:
        for _ in range(96):
            obs = env.reset()
            tot_rwd = 0
            
            for k in trange(100, desc="Episode Steps"):
                obs, rwd, done, _ = env.step()
                env.render()
                tot_rwd += sum(rwd)
                if any(done): break

            print(f"TOTAL REWARD: {tot_rwd}")
            rwd_lst.append(tot_rwd)
            consensus_lst.append(deepcopy(env.consensus_steps))
            obj_pickup_rate.append(env.env.obj_pickup_rate())

        print(f"AVERAGE REWARD: {np.mean(rwd_lst)}")
        print(f"STD DEV: {np.std(rwd_lst)}")
        consensus_lst = np.concatenate(consensus_lst).flatten()
        results = {
            "avg": np.mean(rwd_lst),
            "std": np.std(rwd_lst),
            "min": np.min(rwd_lst),
            "max": np.max(rwd_lst),
            "consensus_avg": np.mean(consensus_lst),
            "consensus_std": np.std(consensus_lst),
        }
        for i in range(len(obj_pickup_rate[0])):
            results[f"pickup_rate_{i+1}"] = np.mean([x[i] for x in obj_pickup_rate])
        return results

    except KeyboardInterrupt:
        env.close()


def train(config=None, default=False):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = DEFAULT_CONFIG if default else wandb.config
        try:
            np.random.seed(config["seed"])
        except:
            pass
        if IBEX: run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]

        # Initialize environment
        env = env_fn(config)

        ############################
        # Start computation
        market = PCFA(env, config)
        results = run_PCFA(market)
        run.summary["test_return_mean"] = results['avg']
        run.summary["test_return_std"] = results['std']
        run.summary["test_return_min"] = results['min']
        run.summary["test_return_max"] = results['max']
        run.summary["consensus_avg"] = results["consensus_avg"]
        run.summary["consensus_std"] = results["consensus_std"]
        # run.summary["revaluation_rate"] = results["revaluation_rate"]
        ############################

        

def env_fn(config):
    kwargs={
            "sz": (config["gymrt2a.size"], config["gymrt2a.size"]),
            "n_agents": config["gymrt2a.N_agents"],
            "n_obj": config["gymrt2a.N_obj"],
            "render": False,
            # "render": True,
            "comm": config["gymrt2a.N_comm"],
            "view_range": config["gymrt2a.view_range"],
            "comm_range": config["gymrt2a.comm_range"],
            "Lsec": config["gymrt2a.Lsec"],
            "one_hot_obj_lvl": True,
            "obj_lvl_rwd_exp": config["gymrt2a.obj_lvl_rwd_exp"],
            "max_obj_lvl": 3,
            "action_grid": config["gymrt2a.action_grid"],
            "share_intention": config["gymrt2a.share_intention"],
            "respawn": config["gymrt2a.respawn"],
            "view_self": True,
        }
    return gymrt2a.env.GridWorldEnv(**kwargs)

if __name__ == "__main__":
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    parser.add_argument("-c", "--count", type=int, default=0, help="Run count")
    try:
        args = parser.parse_args()
        default_config = False
    except:
        args = parser.parse_args(["kaust_visiting_student/pcfa/ud1l5npj"])
        default_config = False
    run_count = args.count if args.count > 0 else None

    sweep_id = args.wandb_sweep
    online = args.online
    wandb.agent(sweep_id, lambda *args, **kw: train(default=default_config,*args, **kw), count=run_count)