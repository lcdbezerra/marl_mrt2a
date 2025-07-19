# PCFA - Project Manager-oriented Coalition Formation Algorithm

A Python package implementing the Project Manager-oriented Coalition Formation Algorithm (PCFA) for multi-agent task allocation in robotic systems.

**This implementation is based on the market-based task allocation algorithm described in:**

> Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment for Cooperative Timing Missions in Dynamic Environments. *Journal of Intelligent and Robotic Systems*, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x

## Installation

1. Create conda environment:
   ```bash
   conda create -y --name marl_mrt2a python=3.10
   conda activate marl_mrt2a
   ```

2. Install `gymrt2a`.

3. Install PCFA package:
   ```bash
   cd PCFA
   pip install -e .
   ```

<!-- ## Quick Start

Run the example:
```bash
python example.py
``` -->

## Package Structure

```
.
├── PCFA/                # Main package directory
│   ├── __init__.py      # Package initialization and public API
│   ├── pcfa.py         # Main implementation (Agent, Task, CommManager, Pathfinder, PCFA classes)
└── example.py          # Example usage script (outside package)
└── setup.py            # Package installation configuration
```
  
  ## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment for Cooperative Timing Missions in Dynamic Environments. *Journal of Intelligent and Robotic Systems*, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x

```bibtex
@article{Oh2017,
   author = {Gyeongtaek Oh and Youdan Kim and Jaemyung Ahn and Han Lim Choi},
   doi = {10.1007/S10846-017-0493-X/METRICS},
   issn = {15730409},
   issue = {1},
   journal = {Journal of Intelligent and Robotic Systems: Theory and Applications},
   keywords = {Cooperative timing mission,Dynamic environment,Market-based task allocation,Multi UAV},
   month = {7},
   pages = {97-123},
   publisher = {Springer Netherlands},
   title = {Market-Based Task Assignment for Cooperative Timing Missions in Dynamic Environments},
   volume = {87},
   url = {https://link.springer.com/article/10.1007/s10846-017-0493-x},
   year = {2017}
}
```