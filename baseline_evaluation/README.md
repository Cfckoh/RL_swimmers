# Baseline Evaluation
Scripts for comparing the analytic baseline to the observed rewards. The analytics baseline is derived with certain assumptions and is not well formed for all parameters. For more on the derivation refer to the paper.


## Config Files
In baseline_config_files there is a interactive python notebook for generating a config file with given parameters. Config files are files of the type .yaml. Data is saved with the name of the config file they were run on. For this reason I recomend not changing config files directly but utilizing the notebook to generate config files of unique names so that runs are not mislabled. 


## Comparing to Baseline 

Run (abc/synth)_baseline_comparison.ipynb to see the figures comparing the baselines in abc/batchelor flows. NOTE: If your python environment is not named stableBase3 you will need to update the bash scripts.