#!/bin/bash

mkdir -p singularity/cache
mkdir -p singularity/tmp

export SINGULARITY_CACHEDIR=$(pwd)/singularity/cache
export APPTAINER_CACHEDIR=$(pwd)/singularity/cache
export SINGULARITY_TMPDIR=$(pwd)/singularity/tmp
export APPTAINER_TMPDIR=$(pwd)/singularity/tmp

singularity_loc=$(which singularity)
# build base container
"$singularity_loc" build base.sif base.def
# build intermediate container
"$singularity_loc" build conda_base_alphafold.sif conda_base_alphafold.def
# build package-specific container
"$singularity_loc" build alphafold.sif alphafold.def

