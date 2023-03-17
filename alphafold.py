#!/usr/bin/env python3.8

import argparse
import csv
import dataclasses
import datetime
import os
import subprocess
import sys
from io import StringIO
from os.path import abspath
from typing import Union


@dataclasses.dataclass
class GPUInfo:
    name: str = 'CPU'
    utilization_gpu: Union[int, float] = float('nan')
    memory_free: Union[int, float] = float('nan')
    memory_total: Union[int, float] = float('nan')
    index: Union[int, float] = None
    uuid: str = 'Unknown'


def get_gpu_information() -> GPUInfo:
    try:
        # No condor - fallback to nvidia-smi parsing
        result = subprocess.run(['/usr/bin/nvidia-smi',
                                 '--query-gpu=name,utilization.gpu,memory.free,memory.total,index,uuid',
                                 '--format=csv,nounits,noheader'],
                                check=True, capture_output=True)
        csv_io = StringIO(result.stdout.decode())
        csv_reader = csv.reader(csv_io)
        available_gpus = [GPUInfo(name=_[0],
                                  utilization_gpu=int(_[1]),
                                  memory_free=int(_[2]),
                                  memory_total=int(_[3]),
                                  index=int(_[4]),
                                  uuid=_[5].strip()) for _ in csv_reader]
        if len(available_gpus) == 0:
            available_gpus.append(GPUInfo())

        # Check that they have a GPU (or proceed with CPU if override turned on)
        try:
            gpu_in_use = os.environ['CUDA_VISIBLE_DEVICES']
            try:
                gpu_in_use = int(gpu_in_use)
            except ValueError:
                # Need to look up the GPU by UUID
                for gpu in available_gpus:
                    if gpu.uuid.startswith(gpu_in_use):
                        return gpu
                raise ValueError(f'The GPU specified by CUDA_VISIBLE_DEVICES ({gpu_in_use}) doesn\'t appear to exist '
                                 f'in the output of nvidia-smi. This may be a system misconfiguration issue.')
        except KeyError:
            return available_gpus[0]

    except (subprocess.CalledProcessError, FileNotFoundError):
        return GPUInfo()


def read_fasta(file_path):
    with open(file_path, 'r') as fp:
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    yield name, ''.join(seq)
                name, seq = line, []
            else:
                seq.append(line)
        if name:
            yield name, ''.join(seq)


def run(arguments):
    # Ensure output directory is writeable
    try:
        test_path = os.path.join(args.output, '.KD5cpxqYzBNqaBZ66guDuh33ns7JYz2jrKq')
        with open(test_path, 'w'):
            pass
        os.unlink(test_path)
    except (IOError, PermissionError):
        raise IOError(f"Your specified output directory '{args.output}' is not "
                      f"writeable. Please choose a different output directory.")

    gpu_info = get_gpu_information()
    if not arguments.force and gpu_info.name == 'CPU':
        print('AlphaFold requires a GPU and none was detected - please re-run on a machine with a GPU.')
        sys.exit(2)

    # Check that there is enough free GPU RAM to run
    if not arguments.force and gpu_info.memory_free < 10000:
        print('AlphaFold requires a large amount of GPU ram available, and this machine has less than 10GB free. '
              'Most likely this means that another user is already using this GPU. Please try again on another '
              'machine. Available machines: https://nmrbox.org/hardware#details')
        sys.exit(3)

    # Only use GPU relaxation on the A100
    if arguments.gpu_relax is None:
        arguments.gpu_relax = gpu_info.name in ['A100-PCIE-40GB', 'NVIDIA A100-PCIE-40GB',
                                                'NVIDIA A100-SXM4-40GB', 'Tesla V100-PCIE-32GB']

    # Print run configuration
    if arguments.verbose:
        print(f"Detected GPU: {gpu_info.name} ({gpu_info.memory_free}MB free GPU RAM out of"
              f" {gpu_info.memory_total}MB total)")
        print(f"GPU relax setting: {args.gpu_relax}")

    sequences = list(read_fasta(arguments.FASTA_file))
    num_chains = len(sequences)

    if num_chains == 0:
        raise ValueError('Your FASTA file does\'t appear to be valid. Please consult documentation here: '
                         'https://en.wikipedia.org/wiki/FASTA_format')
    for chain in sequences:
        if chain[1].upper() != chain[1]:
            raise ValueError('Your FASTA file does\'t appear to be valid. All residues must be specified '
                             f'using capital letters. Problem in sequence: {chain[1]}')
        if 'X' in chain[1]:
            raise ValueError('You have an unknown residue in your sequence - that isn\'t allowed. Problem in sequence '
                             f'{chain[1]}')

    total_aa = sum([len(_[1]) for _ in sequences])
    if total_aa > 800:
        print('AlphaFold uses memory in accordance with the total number of amino acids in all chains. '
              f'As you have more than 800 amino acids in total ({total_aa}), you may run out of memory when running '
              f'AlphaFold. You can look at the VM dashboard in your user profile and select a machine with high '
              f'amounts of memory to try again, but ultimately very long sequences require more RAM than available '
              f'on any NMRbox machine. Furthermore, GPU memory may also be exhausted - if that happens, please try '
              f'rerunning on a machine with an A100 GPU which has 40GB of GPU RAM rather than the 15GB available in '
              f'the T4 machines. For machine details, please see: https://nmrbox.org/hardware#details')

    # Build the basics of the command
    command = ['singularity', 'exec', '--nv', '-B', arguments.database, '-B',
               arguments.output, '-B', arguments.FASTA_file]
    # Determine the template directory path - and add it to the singularity mounts if necessary
    if not arguments.template_mmcif_dir:
        arguments.template_mmcif_dir = os.path.join(arguments.database, '/pdb_mmcif/mmcif_files')
    else:
        command.extend(['-B', arguments.template_mmcif_dir])
    if arguments.custom_config_file:
        command.extend(['--bind', f"{arguments.custom_config_file}:/opt/alphafold/alphafold/model/config.py"])

    # Finish the singularity arguments. Beyond this line are the pass-through AlphaFold arguments.
    command.extend([arguments.singularity_container, '/opt/launcher.sh'])

    # Now add in things common to monomers and multimers
    command.extend([
        '--data_dir', arguments.database,
        '--fasta_paths', arguments.FASTA_file,
        '--output_dir', arguments.output,
        '--max_template_date', arguments.max_template_date,
        '--template_mmcif_dir', arguments.template_mmcif_dir,
        '--obsolete_pdbs_path', os.path.join(arguments.database, '/pdb_mmcif/obsolete.dat'),
        '--mgnify_database_path', os.path.join(arguments.database, '/mgnify/mgy_clusters_2022_05.fa'),
        '--uniref30_database_path', os.path.join(arguments.database, '/uniref30/'),
        '--uniref90_database_path', os.path.join(arguments.database, '/uniref90/uniref90.fasta'),
        '--bfd_database_path', os.path.join(arguments.database, '/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
        "--use_gpu_relax" if arguments.gpu_relax else "--nouse_gpu_relax",
        "--use_precomputed_msas" if arguments.use_precomputed_msas else "--nouse_precomputed_msas"
        ])

    # Add monomer/multimer specific options
    if num_chains == 1:
        print('Found FASTA file with one sequence, treating as a monomer.')
        command.extend(['--pdb70_database_path', os.path.join(arguments.database, '/pdb70/pdb70')])
    elif num_chains > 1:
        print(f'Found FASTA file with {num_chains} sequences, treating as a multimer.')
        command.extend(['--pdb_seqres_database_path', os.path.join(arguments.database, '/pdb_seqres/pdb_seqres.txt'),
                        '--uniprot_database_path',  os.path.join(arguments.database, '/uniprot/uniprot.fasta'),
                        "--num_multimer_predictions_per_model", arguments.num_multimer_predictions_per_model,
                        '--model_preset', 'multimer'])

    print(f'Running AlphaFold, this will take a long time.')
    if arguments.verbose:
        print(f'Executing command: {" ".join(command)}')
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        print(f"AlphaFold raised an exception."
              f"{err.output.decode()}\n\nstderr:\n{err.stderr.decode()}")
        sys.exit(1)

    if arguments.verbose:
        print(f"AlphaFold run stdout:\n{result.stdout.decode()}\n\nstderr:{result.stderr.decode()}")

    print(f"AlphaFold completed without exception. You can find your results in {abspath(arguments.output)}")


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", "-o", action="store", default=".", dest='output',
                    help='The path where the output data should be stored. Defaults to the current directory.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Be verbose.")
parser.add_argument('--version', action='version', version='%(prog)s 2.2.0')
parser.add_argument('FASTA_file', action="store", help='The FASTA file to use for the calculation.')

advanced = parser.add_argument_group('advanced options')
advanced.add_argument("--custom_config_file", action="store",
                      help='Completely replace the standard AlphaFold run configuration file with your own configuration file.'
                           'Provide a path to a configuration file to use rather than the standard one.')
advanced.add_argument('--gpu-relax', dest='gpu_relax', action='store_true', help=argparse.SUPPRESS)
advanced.add_argument('--no-gpu-relax', dest='gpu_relax', action='store_false', help=argparse.SUPPRESS)
advanced.set_defaults(gpu_relax=None)
advanced.add_argument("--database", "-d", action="store", default="/reboxitory/data/alphafold/2.3.1",
                      help='The path to the AlphaFold database to use for the calculation.')
advanced.add_argument("--singularity-container", action="store",
                      default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'alphafold.sif'),
                      help=argparse.SUPPRESS)
advanced.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                      help='Try to run even if no GPU is detected, or the memory is deemed insufficient. Not '
                           'recommended, as either failure or extremely long run times are expected.')

alphafold_group = parser.add_argument_group('advanced AlphaFold pass-through options')
alphafold_group.add_argument("--max_template_date", action="store", default=str(datetime.date.today()),
                             help='If you are predicting the structure of a protein that is already in PDB'
                                  ' and you wish to avoid using it as a template, then max-template-date must be set to'
                                  ' be before the release date of the structure.')
alphafold_group.add_argument("--template_mmcif_dir", action="store",
                             help='Specify a template directory to use instead of the standard mmcif template directory.')
alphafold_group.add_argument("--use_precomputed_msas", action="store_true", default=False,
                             help='Whether to read MSAs that have been written to disk instead of running the MSA '
                                  'tools. The MSA files are looked up in the output directory, so it must stay the same between multiple '
                                  'runs that are to reuse the MSAs. WARNING: This will not check if the sequence, database or configuration have '
                                  'changed. You are recommended to specify an output directory if using this argument to avoid conflicts.')
alphafold_group.add_argument("--num_multimer_predictions_per_model", action="store", default=5,
                             help='How many predictions (each with a different random seed) will be generated per model. E.g. if this is 2 '
                                  'and there are 5 models then there will be 10 predictions per input. '
                                  'Note: this FLAG only applies if your input file is a multimer.')

args = parser.parse_args()

# Get absolute paths
args.database = abspath(args.database)
args.output = abspath(args.output)
args.FASTA_file = abspath(args.FASTA_file)
args.singularity_container = abspath(args.singularity_container)
if args.custom_config_file:
    args.custom_config_file = abspath(args.custom_config_file)

run(args)
