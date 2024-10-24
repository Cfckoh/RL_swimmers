# Actor Physicicsts RL_swimmers
Work done at the University of Arizona exploring control of RL swimmers in Batchelor and ABC flows. Archive link: (coming soon)

## Requirements
NOTE: There is a known issue with the environment as the versions of gym and stable-baselines3 are depreciated and no longer installable with up to date versions of pip. The implemented actor-physcists can be trained and tested without these packages. However, comparison to the out of box stable-baseline3 RL implementations are not possible without these pacakges.

Python conda environment included in environment.yml. The following command will create a python conda environment named stableBase3.

```
conda env create -f py_environment.yml
```

Python code also interacts with Julia where the differential equation solvers are implemented. For julia ensure you have DifferentialEquations, LinearAlgebra, Random, and PyCall

## Outline
This project contains three directories each with there own README. See a directory specfic README for instructions on runing the code. A high level descriptiion of each directory is given below:

- baseline_evaluation: contains scripts for empircally comparing the analytic baseline to observed values in abc and batchelor flows for different parameters
- RL_implmentation: contains scripts for training our actor physicists models in batchelor and abc flows for different parameters


NOTE: Directories contain folders intended for saving figures, csvs, or trained models. These are empty but the appropriate files are saved to them when runing the scripts.
