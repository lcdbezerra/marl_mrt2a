# GyMRT²A Environment

## Installation

- Make sure Conda is installed
- Create a Conda environment named `marl_mrt2a`: `conda create --name marl_mrt2a python=3.10`
- Activate the environment: `conda activate marl_mrt2a`
- Execute the following commands:

```
conda install pip
pip install -e .
pip install -U pygame --user
conda install -c conda-forge libstdcxx-ng
```

- Run an example code to make sure everything is working fine: `python run_hardcoded.py`