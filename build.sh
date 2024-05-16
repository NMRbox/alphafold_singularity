#!/bin/bash

# build base container
sudo /usr/software/bin/singularity build base.sif base.def
# build intermediate container
sudo /usr/software/bin/singularity build conda_base_alphafold.sif conda_base_alphafold.def
# build package-specific container
sudo /usr/software/bin/singularity build alphafold.sif alphafold.def

