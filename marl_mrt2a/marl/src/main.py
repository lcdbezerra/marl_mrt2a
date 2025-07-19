import numpy as np
import os
import random
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import wandb
import traceback

from run import run, run_build

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    # config['run_id'] = wandb.run.id

    # If global path is defined, overwrite results_path
    if config["save_path"]:
        assert os.path.isdir(config["save_path"]), f"Invalid results path was provided: {config['save_path']}"
        results_path = config["save_path"]
    else:
        results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    print(config)
    try:
        map_name = config["env_args"]["key"]
    except:
        map_name = config["env_args"]["map_name"]
    file_obs_path = os.path.join(results_path, f"sacred/{config['name']}/{map_name}")
    print(f"Sacred saving to: {file_obs_path}")

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # run the framework
    run(_run, config, _log)

def main_build(_config, _log):
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    return run_build(None, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)
    
def parse_params(params):
    p_dict = {}
    for p in params:
        p = p.split('=')
        if len(p)!=2: continue
        key, value = p
        try:
            value = eval(value)
        except:
            pass
        # Check for subkeys
        key = key.split('.')
        if len(key)==2:
            if p_dict.get(key[0], None) is None:
                p_dict[key[0]] = {}
            p_dict[key[0]][key[1]] = value
        elif len(key)==1:
            p_dict[key[0]] = value
        else:
            raise RuntimeError(f"Unexpected behavior: {p}, dict more than 2-layers deep?")
    return p_dict

def main_from_arg(argv, just_build=False):
    params = deepcopy(argv)
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]    
    
    # now add all the config to sacred
    ex.add_config(config_dict)
    
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # # Save to disk by default for sacred
    # logger.info("Saving to FileStorageObserver in results/sacred.")
    # file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name}")

    # # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    # ex.observers.append(FileStorageObserver.create(file_obs_path))

    # ex.observers.append(MongoObserver())

    try:
        if just_build:   
            # Add params to config
            new_config = config_copy(config_dict)
            params_dict = parse_params(params)
            new_config = recursive_dict_update(new_config, params_dict)

            return main_build(new_config, logger)
        else:
            ex.run_commandline(params)
            return True
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    main_from_arg(sys.argv)