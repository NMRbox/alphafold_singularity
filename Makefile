.DEFAULT_GOAL := build
.SHELL := /bin/bash
.PHONY := build 

base.sif: base.def
	apptainer --silent build base.sif base.def
	
conda.sif: conda.def base.sif
	apptainer --silent build conda.sif conda.def > conda.build 2>&1

alphafold.sif: alphafold.def conda.sif
	apptainer --silent build alphafold.sif alphafoldc.def > alphafold.build 2>&1

build: alphafold.sif 

clean:
	rm -fr *sif 
	
