#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate alphafold
python3 /opt/alphafold/run_alphafold.py \
 --data_dir "$1" \
 --fasta_paths "$2"  \
 --output_dir "$3" \
 --max_template_date "$4" \
 --model_preset multimer \
 --uniref90_database_path "$1"/uniref90/uniref90.fasta \
 --mgnify_database_path "$1"/mgnify/mgy_clusters_2018_12.fa  \
 --template_mmcif_dir "$1"/pdb_mmcif/mmcif_files \
 --obsolete_pdbs_path "$1"/pdb_mmcif/obsolete.dat \
  --bfd_database_path "$1"/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30_database_path "$1"/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --pdb_seqres_database_path "$1"/pdb_seqres/pdb_seqres.txt \
  --uniprot_database_path "$1"/uniprot/uniprot.fasta \
  "$5"