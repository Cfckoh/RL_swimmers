
## Training Actor Physicist
Training an agent can simply be done from the command line like so:

```
python3 train_abc_AP.py [config_file]
```

Switching train_abc_AP.py with train_batchelor_AP.py will train an agent in batchelor flows.



## Config Files
In confif_files there is a interactive python notebook for generating a config file with given parameters. Config files are files of the type .yaml. Data is saved with the name of the config file they were run on. For this reason I recomend not changing config files directly but utilizing the notebook to generate config files of unique names so that runs are not mislabled. 

## Comparing Agents
abc_agent_comparison.ipynb compares agents of various types including physics agnostic actor critic type methods, fixed phi, and our actor physicists