#!/usr/bin/env python3
"""
Unified script to run all experiments for main comparison.
Supports PCFA (market-based baseline), hardcoded (heuristic baseline),
our model (MAPPO augmented with spatial actions, intention sharing, and task revision),
and ablation studies (tuned down versions of our model).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_wandb_entity():
    """Get the current logged-in wandb entity."""
    try:
        import wandb
        return wandb.api.default_entity
    except Exception as e:
        print(f"⚠️  Warning: Could not get wandb entity: {e}")
        return None


def normalize_sweep_id(sweep_id):
    """
    Normalize sweep ID format to entity/project/sweep_id.
    Accepts:
    - entity/project/sweep_id (full format)
    - project/sweep_id (auto-detects entity)
    - sweep_id (assumes default project and auto-detects entity)
    """
    parts = sweep_id.split('/')
    
    if len(parts) == 3:
        # Already in correct format
        return sweep_id
    elif len(parts) == 2:
        # project/sweep_id format - prepend entity
        entity = get_wandb_entity()
        if entity:
            return f"{entity}/{sweep_id}"
        else:
            raise ValueError("Could not auto-detect wandb entity. Please provide full format: entity/project/sweep_id")
    elif len(parts) == 1:
        # Just sweep_id - use default project and entity
        entity = get_wandb_entity()
        if entity:
            return f"{entity}/marl_mrt2a/{sweep_id}"
        else:
            raise ValueError("Could not auto-detect wandb entity. Please provide full format: entity/project/sweep_id")
    else:
        raise ValueError("Invalid sweep ID format. Expected: entity/project/sweep_id, project/sweep_id, or sweep_id")


def get_pcfa_path():
    """Get the path to the PCFA directory."""
    current_dir = Path(__file__).parent.absolute()
    marl_mrt2a_root = current_dir.parent.parent
    pcfa_path = marl_mrt2a_root / "PCFA"
    
    if not pcfa_path.exists():
        raise FileNotFoundError(f"PCFA directory not found at {pcfa_path}")
    
    return pcfa_path


def get_marl_path():
    """Get the path to the MARL directory."""
    current_dir = Path(__file__).parent.absolute()
    marl_mrt2a_root = current_dir.parent.parent
    marl_path = marl_mrt2a_root / "marl" / "src"
    
    if not marl_path.exists():
        raise FileNotFoundError(f"MARL src directory not found at {marl_path}")
    
    run_script = marl_path / "run_ibex.py"
    if not run_script.exists():
        raise FileNotFoundError(f"MARL run script not found at {run_script}")
    
    return marl_path, run_script


def run_pcfa_sweep(sweep_id, count=None, online=True):
    """Run PCFA (market-based baseline) experiments for the given sweep."""
    
    pcfa_path = get_pcfa_path()
    run_script = pcfa_path / "run.py"
    
    if not run_script.exists():
        raise FileNotFoundError(f"PCFA run script not found at {run_script}")
    
    print(f"🚀 Running PCFA experiments...")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   PCFA path: {pcfa_path}")
    print(f"   Online mode: {online}")
    print(f"   Run count: {count if count else 'all'}")
    
    # Change to PCFA directory
    original_cwd = os.getcwd()
    os.chdir(pcfa_path)
    
    try:
        # Build command
        cmd = ["python", "run.py", sweep_id]
        
        if online:
            cmd.append("-o")
        
        if count:
            cmd.extend(["-c", str(count)])
        
        print(f"   Command: {' '.join(cmd)}")
        print()
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        print(f"✓ PCFA experiments completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ PCFA experiments failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  PCFA experiments interrupted by user")
        return False
    finally:
        # Always return to original directory
        os.chdir(original_cwd)


def run_marl_sweep(sweep_id, count=None, online=True, skip_checks=False):
    """Run our model (MAPPO augmented with spatial actions, intention sharing, and task revision) or ablation studies experiments for the given sweep."""
    
    marl_path, run_script = get_marl_path()
    
    print(f"🚀 Running MARL-based experiments...")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   MARL path: {marl_path}")
    print(f"   Online mode: {online}")
    print(f"   Run count: {count if count else 'all'}")
    
    # Change to MARL directory
    original_cwd = os.getcwd()
    os.chdir(marl_path)
    
    try:
        # Build command
        cmd = ["python", "run_ibex.py", sweep_id]
        
        if online:
            cmd.append("-o")
        
        if count:
            cmd.extend(["-c", str(count)])
        
        print(f"   Command: {' '.join(cmd)}")
        print()
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        print(f"✓ MARL-based experiments completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ MARL-based experiments failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  MARL-based experiments interrupted by user")
        return False
    finally:
        # Always return to original directory
        os.chdir(original_cwd)


def check_conda_environment_pcfa():
    """Check if the proper conda environment is activated for PCFA."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    
    if conda_env == 'pcfa':
        print(f"✓ Detected PCFA conda environment: {conda_env}")
        return True
    else:
        print(f"⚠️  Warning: PCFA conda environment not detected.")
        print(f"   Current environment: {conda_env}")
        print(f"   Make sure to run: conda activate pcfa")
        print(f"   Or install PCFA dependencies in current environment.")
        print()
        return False


def check_conda_environment_marl():
    """Check if the proper conda environment is activated for MARL."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    
    # Check for likely MARL environment names
    expected_envs = ['marl_mrt2a', 'epymarl', 'pymarl']
    
    if conda_env in expected_envs:
        print(f"✓ Detected conda environment: {conda_env}")
        return True
    else:
        print(f"⚠️  Warning: Expected conda environment not detected.")
        print(f"   Current environment: {conda_env}")
        print(f"   Expected one of: {expected_envs}")
        print(f"   Make sure to activate the correct environment with MARL dependencies.")
        print()
        return False


def check_wandb_dependencies():
    """Check if required dependencies are available."""
    try:
        import wandb
        import torch
        import numpy as np
        print("✓ Key dependencies (wandb, torch, numpy) are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required dependency: {e}")
        print("   Make sure all dependencies are installed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experiments for main comparison")
    parser.add_argument("experiment_type", type=str, 
                       choices=["pcfa", "hardcoded", "marl", "steering"],
                       help="Type of experiment to run")
    parser.add_argument("sweep_id", type=str, 
                       help="W&B sweep ID. Accepts: entity/project/sweep_id, project/sweep_id, or just sweep_id")
    parser.add_argument("-c", "--count", type=int, 
                       help="Number of runs to execute (default: all)")
    parser.add_argument("--offline", action="store_true",
                       help="Run in offline mode (don't upload to W&B)")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip environment and dependency checks")
    
    args = parser.parse_args()
    
    # Normalize and validate sweep ID format
    try:
        args.sweep_id = normalize_sweep_id(args.sweep_id)
        print(f"📋 Using sweep ID: {args.sweep_id}")
    except ValueError as e:
        print(f"✗ {e}")
        sys.exit(1)
    
    # Perform environment checks based on experiment type
    if not args.skip_checks:
        if args.experiment_type == "pcfa":
            check_conda_environment_pcfa()
        else:  # MARL experiments
            check_conda_environment_marl()
            if not check_wandb_dependencies():
                print("   Use --skip-checks to bypass this check if you're sure dependencies are correct.")
                sys.exit(1)
    
    # Run experiments based on type
    success = False
    if args.experiment_type == "pcfa":
        success = run_pcfa_sweep(
            sweep_id=args.sweep_id,
            count=args.count,
            online=not args.offline
        )
    else:  # MARL experiments (hardcoded, marl, steering)
        success = run_marl_sweep(
            sweep_id=args.sweep_id,
            count=args.count,
            online=not args.offline,
            skip_checks=args.skip_checks
        )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 