# Main Comparison

This directory contains scripts to run comprehensive comparison experiments between the following:

- **PCFA**: Market-based baseline
- **Our Model**: MAPPO augmented with spatial actions, intention sharing, and task revision
- **Ablation studies**: Tuned down versions of our model

## Usage

### 1. Environment Setup

Make sure the main environment is set up.

```bash
conda activate marl_mrt2a
```

### 2. Create sweeps

We use [Weights&Biases](wandb.ai) for experiment tracking. Make sure it is installed and that your account is set up by running:

```bash
pip install wandb
wandb login
```

Create experiment sweeps for all approaches using the unified script:

```bash
# Create all sweeps (PCFA, spatial actions, steering actions)
python create_sweeps.py --sweep all

# Create specific sweeps
python create_sweeps.py --sweep pcfa         # PCFA baseline only
python create_sweeps.py --sweep marl        # Spatial actions only  
python create_sweeps.py --sweep steering    # Steering actions only

# Quick mode with fewer parameter combinations
python create_sweeps.py --sweep all --quick

# Quick mode with fewer parameter combinations and shorter training (MARL only)
python create_sweeps.py --sweep all --quick --quick-train

# Include ablation studies in MARL sweep (disabled by default)
# Ablations test: intention sharing (on/off) and target revision (on/off)
python create_sweeps.py --sweep marl --include-ablations

# With custom W&B project
python create_sweeps.py --sweep all --project "my_custom_project"
```

After creating sweeps, note the sweep URLs and IDs printed to console for the next step. The script will output commands like:
```
pcfa: python run_experiments.py pcfa marl_mrt2a/[SWEEP_ID]
```

### 3. Run experiments

Run experiments using the sweep IDs obtained from step 2 with the unified script:

```bash
# Run experiments - specify experiment type and sweep ID
python run_experiments.py <experiment_type> <project>/<sweep_id>

# Examples with sweep IDs (project name will be shown in step 2 output)
python run_experiments.py pcfa marl_mrt2a/[SWEEP ID]
python run_experiments.py marl marl_mrt2a/[SWEEP ID]
python run_experiments.py steering marl_mrt2a/[SWEEP ID]

# Run limited number of experiments
python run_experiments.py pcfa marl_mrt2a/[SWEEP ID] -c 5

# Run with checks disabled
python run_experiments.py marl marl_mrt2a/[SWEEP ID] --skip-checks

# Run in offline mode
python run_experiments.py pcfa marl_mrt2a/[SWEEP ID] --offline
```

**Note**: The experiment type must match the sweep type (pcfa, marl, or steering). The sweep URLs and IDs are displayed when creating sweeps in step 2.

### 4. Analyze data

All experiment data is logged to Weights & Biases (wandb). After running experiments, view and analyze results at the corresponding wandb project pages. Use wandb's built-in tools to compare runs, visualize metrics, and export data as needed.
<!-- [TODO] -->

## Environment Configuration

### Parameter Comparison

All experiments use the same base environment configuration for fair comparison:

- **Grid size**: 20x20
- **Number of agents**: 10  
- **Number of objects**: [4, 3, 3] (by level)
- **View range**: 5
- **Object respawn**: Various rates (swept parameter)
- **Seeds**: 10, 20, 30, 40, 50 (5 random seeds)

### Key Differences

| Approach | Action Space | Communication | Training |
|----------|-------------|---------------|----------|
| PCFA | Market-based | High (auction) | None (analytical) |
| MARL (Spatial) | Spatial coordinates | Low (intentions) | 3M timesteps |
| MARL (Steering) | Discrete actions | None | 3M timesteps |

## Files Overview

```
main_comparison/
├── README.md                   # This file
├── create_sweeps.py            # Unified script to create all sweeps
└── run_experiments.py          # Unified script to run all experiments
```

For more details on the overall project structure and installation, see the main [README.md](../../README.md). 