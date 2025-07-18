from setuptools import setup

setup(
    name="gymrt2a",
    version="0.0.1",
    description="Multi-Agent Grid World Gym Environment",
    author="Lucas Bezerra",
    author_email="lcdbezerra@gmail.com",
    url="https://github.com/lcdbezerra/gymrt2a",
    packages=["gymrt2a"],
    install_requires=["numpy", "gym==0.23", "wandb", "importlib-metadata<5", "tqdm", "setuptools==65.5.0", "wheel<0.40.0"],
    license='MIT',
)