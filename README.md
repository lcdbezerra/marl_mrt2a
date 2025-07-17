# Learning Policies for Dynamic Coalition Formation in Multi-Robot Task Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.20397-b31b1b.svg)](https://arxiv.org/abs/2412.20397)

This repository contains the code and simulation environments used in the paper:

**"Learning Policies for Dynamic Coalition Formation in Multi-Robot Task Allocation"**  
*Lucas C. D. Bezerra, Ataíde M. G. dos Santos, and Shinkyu Park*  
Accepted: IEEE Robotics and Automation Letters, 2025


## TL;DR:

This repository contains:

- **GyMRT²A Environment**: a Gym environment for discrete-space, discrete-time MRTA with multi-robot tasks.
- **MARL-MRT²A**: a MAPPO-based algorithm that enables learning of decentralized, low-communication, generalizable task allocation policies for MRT²A; this implementation builds upon the [EPyMARL](https://github.com/uoe-agents/epymarl) codebase.
- **PCFA Baseline**: our implementation of the decentralized market-based approach [PCFA](https://link.springer.com/article/10.1007/s10846-017-0493-x).

To get started, please see the [Installation](#installation) section and run the provided examples.

## Overview

We propose a decentralized, learning-based framework for **dynamic coalition formation** in Multi-Robot Task Allocation (MRTA) under **partial observability**. Our method extends MAPPO with multiple integrated components that allow robots to coordinate and revise task assignments in dynamic, partially observable environments.

### Key Components

- **Spatial Action Maps**: Agents select task locations in spatial coordinates, enabling long-horizon task planning.
- **Robot Motion Planning**: Each robot computes a collision-free A* path to the selected task.
- **Intention Sharing**: Robots share decayed path-based intention maps with nearby agents to support coordination.
- **Custom Policy Architecture**: We propose using a U-Net as the policy architecture, but our code supports custom architecture (that can be implemented as a `torch.nn.Sequential`; see [`nn_utils.py`](marl_mrt2a/marl/src/utils/nn_utils.py) for modules that are currently available)


## Environment

We implement our experiments in a custom Gym environment called **GyMRT²A**, which simulates:

- Grid-world task allocation with dynamic task spawns (Bernoulli or instant respawn)
- Partial observability (limited view and communication ranges)
- Multi-level tasks requiring varying coalition sizes
- Motion planning with obstacles and other agents


## Repository Structure

```
marl_mrt2a/
├── marl_mrt2a/
│   ├── env/               # MRT2A environment
│   ├── PCFA/              # Baseline implementation
│   ├── marl/              # Our method's implementation
│   └── examples/          # Reproducible experiments
│       ├── main_comparison/    # Comparison with baseline and ablation studies
│       ├── generalizability/   # Generalization over task settings
│       ├── scalability/        # Scalability (w.r.t number of robots)
│       └── watch_episode/      # Visualization and monitoring
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch
- NumPy
- OpenAI Gym

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lcdbezerra/marl_mrt2a.git
cd marl_mrt2a
```

2. Create a conda environment:
```bash
conda create -n marl_mrt2a python=3.10
conda activate marl_mrt2a
conda install pip -y
```

3. Install the base environment and the baseline (development mode)
```bash
cd env
pip install -e .
cd ../PCFA
pip install -e .
cd ../
```

4. Install MARL dependencies:
```bash
cd marl
pip install wandb
pip install -r requirements.txt
```

## Experiments

Experiments are reproducible through the examples in the `examples/` directory:

### Main Comparison (`examples/main_comparison/`)
Compare the proposed method against baseline approaches:
- Traditional task allocation methods
- Standard MAPPO
- Other multi-agent learning approaches

<!-- ### 2. Generalizability (`examples/generalizability/`)
Test the framework's ability to generalize across:
- Different task types
- Varying environment configurations
- Diverse robot capabilities

### 3. Scalability (`examples/scalability/`)
Evaluate performance with:
- Different robot population sizes
- Varying task loads
- Large-scale scenarios

### 4. Visualization (`examples/watch_episode/`)
Monitor and visualize:
- Robot movements and coalitions
- Task allocation dynamics
- Learning progress -->

### Running Experiments

```bash
# Run main comparison experiments
python examples/main_comparison/run_comparison.py
```
<!-- ```bash
# Run main comparison experiments
python examples/main_comparison/run_comparison.py

# Run scalability experiments
python examples/scalability/run_scalability.py

# Run generalizability experiments
python examples/generalizability/run_generalizability.py

# Watch trained agents
python examples/watch_episode/watch_episode.py
``` -->

## Citation

If you use this code, please cite:

```bibtex
@misc{bezerra2025learningdcfmrta,
      title={Learning Policies for Dynamic Coalition Formation in Multi-Robot Task Allocation}, 
      author={Lucas C. D. Bezerra and Ataíde M. G. dos Santos and Shinkyu Park},
      year={2025},
      eprint={2412.20397},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.20397}, 
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.