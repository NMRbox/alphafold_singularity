# AlphaFold Singularity Container

This repo provides definition files to build a singularity container of AlphaFold v2 
(https://github.com/deepmind/alphafold) that will run and be easy to invoke in the NMRbox
environment.

Build instructions from [non-docker setting](https://github.com/kalininalab/alphafold_non_docker) by kalininalab were used.

## Build container
```
# build base container
sudo /usr/software/singularity/bin/singularity build base.sif base.def
# build intermediate container
sudo /usr/software/singularity/bin/singularity build conda_base_alphafold.sif conda_base_alphafold.def
# build package-specific container
sudo /usr/software/singularity/bin/singularity build alphafold.sif alphafold.def
```

## Run AlphaFold
```
./alphafold.py /path/to/fasta/file
```

### Notes

* The alphafold.py run script has no requirements and should run in vanilla python 3.8.
* The run script allows customizing the database location and max_template_date. Call with `-h` to see usage information.
* By default, this uses the `monomer` model for monomers and the `multimer` model for multimers,
  and uses the `full_dbs` option for better quality results. For more details, see https://github.com/deepmind/alphafold#running-alphafold

## AlphaFold 3

`alphafold3/alphafold3.py` is the AlphaFold 3 equivalent of `alphafold.py`. It runs the AlphaFold 3 container
(built from the official AlphaFold 3 Dockerfile) and accepts a FASTA file of protein sequences (converted
automatically to an AlphaFold 3 input JSON), an AlphaFold 3 input JSON file, or a directory of JSON files.

```
./alphafold3.py /path/to/input.fasta
./alphafold3.py /path/to/input.json
```

When deployed, the script lives in the AlphaFold 3 database directory, next to `alphafold3.sif` and a `models`
directory containing the model parameters. For testing, copy `alphafold3/alphafold3.ini.example` to
`alphafold3/alphafold3.ini` (or pass `--config`) to point at a different container, database, and model directory.
Run with `-h` for all options.

The script picks GPU-appropriate settings automatically: unified memory on GPUs with less than 80GB of memory, and
the XLA attention implementation (plus the required `XLA_FLAGS`) on pre-Ampere GPUs such as the V100.
