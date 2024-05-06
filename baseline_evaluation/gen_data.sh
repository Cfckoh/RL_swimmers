#!/bin/bash

# Activate Conda environment
source activate stableBase3

# Run script with argument
python3 abc_baseline_eval.py "$1"

# Deactivate Conda environment
conda deactivate