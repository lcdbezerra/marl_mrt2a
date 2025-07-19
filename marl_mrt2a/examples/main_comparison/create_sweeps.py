#!/usr/bin/env python3
"""
Unified script to create all sweeps for main comparison experiments.
Supports PCFA (market-based baseline), hardcoded (heuristic baseline), 
our model (MAPPO augmented with spatial actions, intention sharing, and task revision),
and ablation studies (tuned down versions of our model).
"""

import wandb
import numpy as np
import argparse
import sys
import os


# Common parameters shared across all sweep types
CONST_PARAMS = {
    "gymrt2a.size": {"value": 20},
    "gymrt2a.Lsec": {"value": 2},
    "gymrt2a.N_agents": {"value": 10},
    "gymrt2a.N_obj": {"value": [4, 3, 3]},
    "gymrt2a.obj_lvl_rwd_exp": {"value": 2},
    "gymrt2a.view_self": {"value": False},
}

# Common MARL-specific parameters
MARL_CONST_PARAMS = {
    **CONST_PARAMS,
    "gymrt2a.N_comm": {"value": 0},
    "gymrt2a.comm_range": {"value": 8},
    "gymrt2a.action_grid": {"value": True},
    "env_args.hardcoded": {"value": False},
    "env_args.comm_rounds": {"value": 0},
    "current_target_factor": {"value": 1},
    "config": {"value": "mappo"},
    "env_config": {"value": "gridworld"},
    "critic_type": {"value": "cnn_cv_critic"},
    "agent": {"value": "cnn"},
    "agent_distance_exp": {"value": 1},
    "gamma": {"value": 0.95},
    "standardise_returns": {"value": False},
    "q_nstep": {"value": 5},
    "obs_agent_id": {"value": False},
    "buffer_size": {"value": 10},
    "test_nepisode": {"value": 96},
    "target_update_interval_or_tau": {"value": 200},
}


def create_pcfa_sweep(project="marl_mrt2a", quick_mode=False):
    """Create a PCFA (market-based baseline) sweep configuration for main comparison."""
    
    sweep_config = {
        "name": "PAPER: PCFA (Market-based Baseline)" + (" (Quick)" if quick_mode else ""),
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "test_return_mean",
        },
        "parameters": {
            **CONST_PARAMS,
            "gymrt2a.N_comm": {"value": 4},
            "gymrt2a.view_range": {"value": 5},
            "gymrt2a.comm_range": {"values": [20, 8] if not quick_mode else [8]},
            "gymrt2a.action_grid": {"value": True},
            "gymrt2a.share_intention": {"value": False},
            "gymrt2a.respawn": {
                "values": ([True] + list(np.arange(.01, .041, .01))) if not quick_mode else [True, .02]
            },
            "env_args.hardcoded": {"value": "pcfa"},
            "seed": {"values": [10, 20, 30, 40, 50] if not quick_mode else [10, 20]},
            "worth_exponent": {"value": 2},
            "max_task_sequence_length": {"value": 100},
            "market_loop_length": {"value": True},
            "neighbors_topology": {"value": False},
            "partial_observability": {"value": True},
            "lambda": {"values": [1.5, 2, 2.5, 3.] if not quick_mode else [2, 2.5]},
            "test_nepisode": {"value": 8 if quick_mode else 96},
        },
    }
    
    print("Creating PCFA sweep...")
    print(f"Project: {project}")
    print(f"Quick mode: {quick_mode}")
    
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"✓ PCFA sweep created with ID: {sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"✗ Failed to create PCFA sweep: {e}")
        return None


def create_hardcoded_sweep(project="marl_mrt2a", quick_mode=False):
    """Create a hardcoded (heuristic baseline) sweep configuration for main comparison."""
    
    sweep_config = {
        "name": "PAPER: Hardcoded (Heuristic Baseline)" + (" (Quick)" if quick_mode else ""),
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "best_test_return_mean",
        },
        "parameters": {
            **CONST_PARAMS,
            "gymrt2a.N_comm": {"value": 2},
            "gymrt2a.view_range": {"value": 5},
            "gymrt2a.comm_range": {"value": 8},
            "gymrt2a.action_grid": {"value": True},
            "gymrt2a.share_intention": {"value": "target"},
            "gymrt2a.respawn": {"values": [True, .01, .02, .03, .04] if not quick_mode else [True]},
            "env_args.hardcoded": {"value": True},
            "agent_reeval_rate": {"values": True},
            "seed": {"values": [10, 20, 30, 40, 50] if not quick_mode else [10]},
            "filter_avail_by_objects": {"value": True},
            "test_nepisode": {"value": 8 if quick_mode else 96},
        },
    }
    
    print("Creating hardcoded baseline sweep...")
    print(f"Project: {project}")
    print(f"Quick mode: {quick_mode}")
    
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"✓ Hardcoded sweep created with ID: {sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"✗ Failed to create hardcoded sweep: {e}")
        return None


def create_marl_sweep(project="marl_mrt2a", quick_mode=False, quick_train=False):
    """Create our model (MAPPO augmented with spatial actions, intention sharing, and task revision) sweep configuration for main comparison."""
    
    sweep_config = {
        "name": "PAPER: Our Model (MAPPO + Spatial Actions + Intention Sharing + Task Revision)" + (" (Quick)" if quick_mode else "") + (" (Fast Train)" if quick_train else ""),
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "best_test_return_mean",
        },
        "parameters": {
            **MARL_CONST_PARAMS,
            "critic_arch": {"value": "unet,4,1,2&batchNorm1d;linear,25;relu" if quick_train else "unet,8,1,2&batchNorm1d;linear,50;relu"},
            "agent_arch": {"value": "unet,4,1,2&" if quick_train else "unet,8,1,2&"},
            "gymrt2a.view_range": {"values": [3, 5, 8] if not quick_mode else [5]},
            "action_grid": {"value": True},
            "gymrt2a.share_intention": {
                "values": ["target", "path", "channel", False] if not quick_mode else ["path", False]
            },
            "gymrt2a.respawn": {
                "values": [True, .01, .02, .03, .04] if not quick_mode else [True, .02]
            },
            "seed": {"values": [10, 20, 30, 40, 50] if not quick_mode else [10, 20]},
            "agent_reeval_rate": {"values": [True, 0] if not quick_mode else [True]},
            "filter_avail_by_objects": {"value": True},
            "t_max": {"value": 100_000 if quick_train else 4_000_000},
            "test_nepisode": {"value": 8 if quick_train else 96},
        },
    }
    
    print("Creating our model (MAPPO augmented with spatial actions, intention sharing, and task revision) sweep...")
    print(f"Project: {project}")
    print(f"Quick mode: {quick_mode}")
    print(f"Quick train: {quick_train}")
    
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"✓ Our model sweep created with ID: {sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"✗ Failed to create our model sweep: {e}")
        return None


def create_steering_actions_sweep(project="marl_mrt2a", quick_mode=False, quick_train=False):
    """Create ablation studies (tuned down versions of our model) sweep configuration for main comparison."""
    
    sweep_config = {
        "name": "PAPER: Ablation Studies (Tuned Down Versions of Our Model)" + (" (Quick)" if quick_mode else "") + (" (Fast Train)" if quick_train else ""),
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "best_test_return_mean",
        },
        "parameters": {
            **MARL_CONST_PARAMS,
            "critic_arch": {"value": "unet,4,1,2&batchNorm1d;linear,25;relu" if quick_train else "unet,8,1,2&batchNorm1d;linear,50;relu"},
            "agent_arch": {"value": "unet,4,1,2&" if quick_train else "unet,8,1,2&"},
            "gymrt2a.view_range": {"values": [3, 5, 8] if not quick_mode else [5]},
            "action_grid": {"value": False},  # Key difference - steering actions
            "gymrt2a.share_intention": {"value": False},
            "gymrt2a.respawn": {
                "values": [True, .01, .02, .03, .04] if not quick_mode else [True, .02]
            },
            "seed": {"values": [10, 20, 30, 40, 50] if not quick_mode else [10, 20]},
            "agent_reeval_rate": {"value": 0},
            "filter_avail_by_objects": {"value": False},
            "t_max": {"value": 100_000 if quick_train else 4_000_000},
            "test_nepisode": {"value": 8 if quick_train else 96},
        },
    }
    
    print("Creating ablation studies (tuned down versions of our model) sweep...")
    print(f"Project: {project}")
    print(f"Quick mode: {quick_mode}")
    print(f"Quick train: {quick_train}")
    
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"✓ Ablation studies sweep created with ID: {sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"✗ Failed to create ablation studies sweep: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create all sweeps for main comparison")
    parser.add_argument("--sweep", type=str, 
                       choices=["pcfa", "hardcoded", "marl", "steering", "all"], 
                       default="all", help="Which sweep(s) to create")
    
    # Common W&B settings
    parser.add_argument("--project", type=str, default="marl_mrt2a", 
                       help="W&B project name (default: marl_mrt2a)")
    
    # Common settings
    parser.add_argument("--quick", action="store_true",
                       help="Use fewer parameter combinations for quick testing")
    parser.add_argument("--quick-train", action="store_true",
                       help="Use shorter training time for MARL sweeps")
    
    args = parser.parse_args()
    
    created_sweeps = []
    
    # Create selected sweeps
    if args.sweep in ["pcfa", "all"]:
        sweep_id = create_pcfa_sweep(
            project=args.project,
            quick_mode=args.quick
        )
        if sweep_id:
            created_sweeps.append(("pcfa", f"{args.project}/{sweep_id}"))
    
    if args.sweep in ["hardcoded", "all"]:
        sweep_id = create_hardcoded_sweep(
            project=args.project,
            quick_mode=args.quick
        )
        if sweep_id:
            created_sweeps.append(("hardcoded", f"{args.project}/{sweep_id}"))
    
    if args.sweep in ["marl", "all"]:
        sweep_id = create_marl_sweep(
            project=args.project,
            quick_mode=args.quick,
            quick_train=args.quick_train
        )
        if sweep_id:
            created_sweeps.append(("marl", f"{args.project}/{sweep_id}"))
    
    if args.sweep in ["steering", "all"]:
        sweep_id = create_steering_actions_sweep(
            project=args.project,
            quick_mode=args.quick,
            quick_train=args.quick_train
        )
        if sweep_id:
            created_sweeps.append(("steering", f"{args.project}/{sweep_id}"))
    
    # Print summary
    if created_sweeps:
        print(f"\n🎯 Created {len(created_sweeps)} sweep(s). Next steps:")
        for sweep_type, sweep_id in created_sweeps:
            print(f"   {sweep_type}: python run_experiments.py {sweep_type} {sweep_id}")
    else:
        print("✗ No sweeps were created successfully")
        sys.exit(1)


if __name__ == "__main__":
    main() 