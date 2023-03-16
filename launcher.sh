#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate alphafold
python3 /opt/alphafold/run_alphafold.py "$@"
