#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate alphafold
python3 -c 'import jax; print(jax.devices())'
