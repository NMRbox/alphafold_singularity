#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate alphafold
export TF_CPP_MIN_LOG_LEVEL=0
python3 -c 'import jax; print(jax.devices())'
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
