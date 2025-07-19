import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import cProfile
import pstats
import io
from tqdm import tqdm
from memory_profiler import profile
import numpy as np
from modules.agents.custom_agent import CNNAgent
from types import SimpleNamespace

DEFAULT_CONFIG = {
    "buffer_size": 10,
    # "config": "vdn",
    "config": "mappo", 
    "critic_type": "cnn_cv_critic",
    "env_config": "gridworld", "agent": "cnn",
    # "agent_arch": "resnet;conv2d,64,1;relu;interpolate,2;conv2d,1,1;relu;interpolate,1.7&",
    # "critic_arch": "resnet&batchNorm1d;linear,128;relu;linear,32;relu",
    "agent_arch": "unet,8,1,2&",
    "critic_arch": "unet,8,1,2&batchNorm1d;linear,50;relu",
    "strategy": "cnn",
    # "strategy": "hardcoded",
    # "env_config": "gymma",
    "hidden_dim": 512,
    "obs_agent_id": False,
    "gymrt2a.Lsec": 2,
    "gymrt2a.N_agents": 10,
    "env_args.hardcoded": "comm",
    "gymrt2a.N_comm": 4,
    "gymrt2a.N_obj": [4, 3, 3],
    "gymrt2a.comm_range": 8,
    # "gymrt2a.size": 40,
    "gymrt2a.size": 20,
    "gymrt2a.view_range": 4,
    "gymrt2a.action_grid": True,
    "gymrt2a.respawn": True,
    "action_grid": True,
    "current_target_factor": None,
    # "gymrt2a.share_intention": "path",
    # "share_intention": "path",
    # "gymrt2a.share_intention": False,
    # "share_intention": False,
    "gymrt2a.share_intention": "channel",
    "share_intention": "channel",
    "seed": 10,
    "t_max": 2_000_000,
    "env_args.curriculum": True,
}

def create_dummy_args():
    """Create dummy arguments for CNNAgent initialization"""
    args = SimpleNamespace()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_actions = 17*17
    args.n_agents = 1
    args.agent_reeval_rate = True
    args.current_target_factor = None
    args.agent_arch = "unet,8,1,2&"
    args.agent_distance_exp = 1.0
    return args

def run_multiple_profiles_impl(num_runs=1000):
    """
    Implementation of profile running without decorators
    """
    # Initialize agent once
    input_shape = (6, 17, 17)  # Example input shape
    args = create_dummy_args()
    agent = CNNAgent(input_shape, args).eval()
    
    # Create dummy input
    batch_size = 1
    
    # Create dummy env_info
    env_info = {
        "robot_info": [
            [(0, 0), (1, 1)]  # (position, command) for each agent
        ]
    }
    
    # Run forward pass multiple times
    inner_loop_size = 5000
    execution_times = [0.0] * (num_runs//inner_loop_size)
    
    print(f"\nRunning forward pass {num_runs} times...")
    for j in tqdm(range(num_runs//inner_loop_size), desc="Batches", unit="batch"):
        input_tensors = [torch.randn(batch_size, *input_shape).to(args.device) for _ in range(inner_loop_size)]
        # Start timing before the loop
        start_time_total = time.time()
        
        for i in range(inner_loop_size):
            # if i % 10 == 0:
            #     print(f"Run {i+1}/{num_runs}")
            
            # Run forward pass without individual timing
            output = agent(input_tensors[i], env_info=env_info)
            # outputs.append(output)
        
        # End timing after the loop
        end_time_total = time.time()
    
        # Calculate average execution time
        total_execution_time = end_time_total - start_time_total
        avg_execution_time = total_execution_time / inner_loop_size
        
        # Collect execution time for this batch
        execution_times[j] = avg_execution_time
    
    # Calculate statistics across all batches
    avg_time = np.mean(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    std_dev = np.std(execution_times)
    
    print("\n" + "="*50)
    print(f"Average execution time: {avg_time:.8f} seconds")
    print(f"Minimum execution time: {min_time:.8f} seconds")
    print(f"Maximum execution time: {max_time:.8f} seconds")
    print(f"Standard deviation: {std_dev:.8f} seconds")
    print("="*50)
    
    return avg_time, min_time, max_time, std_dev

def run_multiple_profiles_impl_2(num_runs=1000):
    """
    Implementation of profile running without decorators
    """
    # Initialize agent once
    input_shape = (6, 17, 17)  # Example input shape
    args = create_dummy_args()
    agent = CNNAgent(input_shape, args).eval()
    
    # Create dummy input
    batch_size = 1
    
    # Create dummy env_info
    env_info = {
        "robot_info": [
            [(0, 0), (1, 1)]  # (position, command) for each agent
        ]
    }
    
    # Run forward pass multiple times
    execution_times = [0.0] * (num_runs)
    
    print(f"\nRunning forward pass {num_runs} times...")
    for j in tqdm(range(num_runs), desc="Batches", unit="batch"):
        input_tensors = torch.randn(batch_size, *input_shape).to(args.device)
        # Start timing before the loop
        start_time_total = time.time()
            
        # Run forward pass without individual timing
        output = agent(input_tensors, env_info=env_info)
        # outputs.append(output)
        
        # End timing after the loop
        end_time_total = time.time()
    
        # Calculate average execution time
        total_execution_time = end_time_total - start_time_total
        avg_execution_time = total_execution_time
        
        # Collect execution time for this batch
        execution_times[j] = avg_execution_time
    
    # Calculate statistics across all batches
    avg_time = np.mean(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    std_dev = np.std(execution_times)
    
    print("\n" + "="*50)
    print(f"Average execution time: {avg_time:.8f} seconds")
    print(f"Minimum execution time: {min_time:.8f} seconds")
    print(f"Maximum execution time: {max_time:.8f} seconds")
    print(f"Standard deviation: {std_dev:.8f} seconds")
    print("="*50)
    
    return avg_time, min_time, max_time, std_dev

# Apply memory profiling decorator conditionally
def run_multiple_profiles(num_runs=1000, use_memory_profiling=False):
    """
    Wrapper function that conditionally applies memory profiling
    """
    if use_memory_profiling:
        # Apply memory profiling decorator dynamically
        profiled_func = profile(run_multiple_profiles_impl)
        return profiled_func(num_runs)
    else:
        # Run without memory profiling
        # return run_multiple_profiles_impl(num_runs)
        return run_multiple_profiles_impl_2(num_runs)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Profile CNNAgent performance')
    
    # Default arguments
    default_runs = 10
    default_memory_profile = False
    default_profile = False
    
    parser.add_argument('--runs', type=int, default=default_runs, 
                        help=f'Number of forward passes to run (default: {default_runs})')
    parser.add_argument('--memory-profile', action='store_true', default=default_memory_profile,
                        help='Enable memory profiling (slower)')
    parser.add_argument('--profile', action='store_true', default=default_profile,
                        help='Enable cProfile')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Run cProfile
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    
    # Run profiling
    avg_time, min_time, max_time, std_dev = run_multiple_profiles(
        num_runs=args.runs, 
        use_memory_profiling=args.memory_profile
    )
    
    if args.profile:
        pr.disable()
        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print("\ncProfile Results:")
        print(s.getvalue())

if __name__ == "__main__":
    main()
    
"""
USAGE:

python profiling/profile_cnn_agent.py --runs 1000 --memory-profile
"""