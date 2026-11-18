# AlphaFold Singularity Container

This repo builds a Singularity container of AlphaFold 3 (https://github.com/google-deepmind/alphafold3) and provides
a run script that makes it easy to invoke in the NMRbox environment.

## Build container
```
./build.sh          # build the latest AlphaFold 3 release
./build.sh v3.0.1   # or a specific tag, branch, or commit
```

`build.sh` clones (or updates) the AlphaFold 3 repository into `alphafold3_src/`, builds the Docker image from its
`docker/Dockerfile` as described in the upstream installation instructions, and converts it to `alphafold3.sif`. It
requires docker or podman (set `CONTAINER_TOOL` to choose) and singularity or apptainer.

The model parameters must be requested from Google separately, see https://github.com/google-deepmind/alphafold3.

## Run AlphaFold 3

`alphafold3.py` runs the AlphaFold 3 container (built from the official AlphaFold 3 Dockerfile). It accepts a FASTA
file of protein sequences (converted automatically to an AlphaFold 3 input JSON), an AlphaFold 3 input JSON file, or a
directory of JSON files.

```
./alphafold3.py /path/to/input.fasta
./alphafold3.py /path/to/input.json
```

When deployed, the script lives in the AlphaFold 3 database directory, next to `alphafold3.sif` and a `models`
directory containing the model parameters. For testing, copy `alphafold3.ini.example` to `alphafold3.ini` (or pass
`--config`) to point at a different container, database, and model directory. Run with `-h` for all options.

The script picks GPU-appropriate settings automatically: unified memory on GPUs with less than 80GB of memory, and
the XLA attention implementation (plus the required `XLA_FLAGS`) on pre-Ampere GPUs such as the V100.
