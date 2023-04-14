# AlphaFold Singularity Container

This repo provides definition files to build a singularity container of AlphaFold v2 
(https://github.com/deepmind/alphafold) that will run and be easy to invoke in the NMRbox
environment.

Build instructions from [non-docker setting](https://github.com/kalininalab/alphafold_non_docker) by kalininalab were used.

## Setup

Clone repository into your home directory and `cd` into the cloned folder. 

## Build container
```
# build base container
apptainer build base.sif base.def
# build alphafold container
apptainer build alphafold.sif alphafold.def
```

## Run AlphaFold Job

Customize or add the following options to a typical CHTC HTCondor submit file: 

```
universe = container
container_image = alphafold.sif
requirements = (HasGpulabData == true)

transfer_executable = false
# replace with multimer.sh if applicable
executable = /opt/alphafold/monomer.sh
arguments = /gpulab_data/alphafold FASTA_file 

transfer_input_files = alphafold.sif, FASTA_file

# request CPUs, GPUS, etc.
```


### Notes

* The alphafold.py run script has no requirements and should run in vanilla python 3.8.
* The run script allows customizing the database location and max_template_date. Call with `-h` to see usage information.
* By default, this uses the `monomer` model for monomers and the `multimer` model for multimers,
  and uses the `full_dbs` option for better quality results. For more details, see https://github.com/deepmind/alphafold#running-alphafold
