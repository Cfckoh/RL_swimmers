# Actor Physicicsts RL_swimmers
Work done at the University of Arizona exploring control of RL swimmers in Batchelor and ABC flows. Archive link: (coming soon)

## Requirements

Python conda environment included in environment.yml. The following command will create a python conda environment named stableBase3.

'''
conda env create -f environment.yml
'''

Python code also interacts with Julia where the differential equation solvers are implemented. To set up the appropriate Julia environment... TODO instructions for Julia

## Outline
This project contains three directories each with there own README. See a directory specfic README for instructions on runing the code. I high level descriptiion of each directory is given below:

- baseline_evaluation: contains scripts for empircally comparing the analytic baseline to observed values in abc and batchelor flows for different parameters
- RL_implmentation: contains scripts for training our actor physicists models in batchelor and abc flows for different parameters
- results_analysis: contains scripts and notebooks for comparing agents and making a few figures to analyze results (TODO)

NOTE: Directories may contain folders intended for saving figures, csvs, or trained models. These are empty in the push but are used by the scripts when run to save appropriate files.  