#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdm

python run_mdm.py "$@"