#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate motiondiffuse

python run_motiondiffuse.py "$@"