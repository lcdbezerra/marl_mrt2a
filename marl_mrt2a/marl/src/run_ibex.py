import os
import wandb
import numpy as np
import argparse
import gymrt2a
from multiprocessing import cpu_count

import gym
from gym.envs.registration import register

IBEX = True if os.path.exists("/ibex/") else False
usr_name = os.environ["USER"]
DSA = usr_name=="ubuntu"
scratch_dir = f"/ibex/user/{usr_name}/runs/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/"
# base_dir = f"/home/{usr_name}/code/epymarl"
# sys.path.append(base_dir)

import main

parser = argparse.ArgumentParser(description="Training Script")

def config2txt(config):
    discard_start_with = "mrt2a"
    keys_to_discard = ["config", "env_config", "save_model", "save_path", "strategy"]
    requires_quotes = ["critic_arch", "agent_arch"]

    comb = []
    for k, v in config.items():
        if k.startswith(discard_start_with) or k in keys_to_discard:
            continue
        if k in requires_quotes:
            comb.append(f'{k}="{v}"')
        else:
            comb.append(f"{k}={v}")
    if comb==[]:
        return ""
    txt = " ".join(comb)
    return txt+" "

def run_hardcoded(env, config):
    env = gymrt2a.env.HardcodedWrapper(env, policy=gymrt2a.policy.HighestLvlObjPolicy, 
                                         agent_reeval_rate=config["agent_reeval_rate"])
    rwd_lst = []
    obj_pickup_rate = []
    try:
        for _ in range(96):
            obs = env.reset()
            tot_rwd = 0
            for k in range(100):
                obs, rwd, done, _ = env.step()
                env.render()
                tot_rwd += sum(rwd)
                if any(done): break

            print(f"TOTAL REWARD: {tot_rwd}")
            rwd_lst.append(tot_rwd)
            obj_pickup_rate.append(env.env.obj_pickup_rate())
            # print(obj_pickup_rate[-1])

        print(f"AVERAGE REWARD: {np.mean(rwd_lst)}")
        print(f"STD DEV: {np.std(rwd_lst)}")
        results = {}
        results["return_mean"] = np.mean(rwd_lst)
        results["return_std"] = np.std(rwd_lst)
        for i in range(len(obj_pickup_rate[0])):
            results[f"pickup_rate_{i+1}"] = np.mean([x[i] for x in obj_pickup_rate])
        return results

    except KeyboardInterrupt:
        env.close()


def train(config=None, default=False, online=False):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = wandb.config
        try:
            np.random.seed(config["seed"])
        except:
            pass
        if IBEX: run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]
        run.summary["username"] = os.environ["USER"]

        # Save path
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define environment key
        if config.get("env_args.curriculum", False):
            env_key = register_curriculum_env(run.id, config)
        else:
            env_key = register_env(run.id, config)
        print(f"Environment: {env_key}")

        if config.get("current_target_factor", None) is not None:
            run.summary["log_current_target_factor"] = np.log(config["current_target_factor"])
        else:
            run.summary["log_current_target_factor"] = None

        # if config["env_args.hardcoded"]==True:
        if config.get("env_args.hardcoded", False) == True:
            results_dict = run_hardcoded(gym.make(env_key), config)
            run.summary["best_test_return_mean"] = results_dict["return_mean"]
            # run.summary["best_test_return_std"] = results_dict["return_std"]
            for k,v in results_dict.items():
                run.summary[k] = v
        else:
            # Define save path
            save_path = scratch_dir + run.id + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Define script to call
            n_parallel = config.get("buffer_size", None)
            if n_parallel is None:
                n_parallel = 16 if IBEX else min(cpu_count()//2, 16)
                # n_parallel = int(os.getenv("SLURM_CPUS_PER_TASK")) if IBEX else min(cpu_count()//2, 16)
            config["batch_size_run"] = n_parallel # add number of parallel envs to config
            txt_args = f'main.py --config={config["config"]} --env-config={config["env_config"]} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
            txt_args += f" runner=parallel batch_size_run={n_parallel}"
            if not IBEX and not DSA:
                txt_args += " use_cuda=False"
            print("python3 " + txt_args)

            # Run EPyMARL training script
            main.main_from_arg(txt_args.split(' '))

def register_env(id, config):
    env_id = f"GridWorld-Custom-{id}-v0"
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
            "view_self": config.get("gymrt2a.view_self", True),
            "max_obj_lvl": 3,
            "action_grid": config["gymrt2a.action_grid"],
            "share_intention": config["gymrt2a.share_intention"],
            "respawn": config["gymrt2a.respawn"],
        }
    print(kwargs)

    register(
        id=env_id,
        entry_point="gymrt2a.env:GridWorldEnv",
        kwargs=kwargs,
        order_enforce=False, # VERY IMPORTANT!!!
    )
    return env_id

def register_curriculum_env(id, config):
    env_id = f"GridWorld-Curriculum-{id}-v0"
    kwargs={
            "render": False,
            "train_args": config["gymrt2a.train"],
            "eval_args": config["gymrt2a.eval"],
        }
    print(kwargs)

    register(
        id=env_id,
        entry_point="gymrt2a.env:CurriculumEnv",
        kwargs=kwargs,
        order_enforce=False, # VERY IMPORTANT!!!
    )
    return env_id

if __name__ == "__main__":
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    parser.add_argument("-c", "--count", type=int, default=0, help="Run count (optional)")
    args = parser.parse_args()
    default_config = False

    sweep_id = args.wandb_sweep
    run_count = args.count if args.count > 0 else None
    online = args.online
    wandb.agent(sweep_id, lambda *args, **kw: train(default=default_config, online=online, *args, **kw), count=run_count)