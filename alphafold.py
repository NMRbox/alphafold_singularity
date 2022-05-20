#!/usr/bin/env python3.8

import argparse
import csv
import datetime
import os
import subprocess
import sys
from io import StringIO
from os.path import abspath
from typing import List


def get_gpu_information() -> List[dict]:
    try:
        # No condor - fallback to nvidia-smi parsing
        result = subprocess.run(['/usr/bin/nvidia-smi',
                                 '--query-gpu=name,utilization.gpu,memory.free,memory.total,index',
                                 '--format=csv,nounits,noheader'],
                                check=True, capture_output=True)
        csv_io = StringIO(result.stdout.decode())
        csv_reader = csv.reader(csv_io)
        return [{'name': _[0],
                 'utilization.gpu': int(_[1]),
                 'memory.free': int(_[2]),
                 'memory.total': int(_[3]),
                 'index': int(_[4])} for _ in csv_reader]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [{'name': 'CPU',
                 'utilization.gpu': float('nan'),
                 'memory.free': float('nan'),
                 'memory.total': float('nan'),
                 'index': None}]


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
        with open(test_path, 'w') as test:
            pass
        os.unlink(test_path)
    except (IOError, PermissionError):
        raise IOError(f"Your specified output directory '{args.output}' is not "
                      f"writeable. Please choose a different output directory.")

    # Check that they have a GPU (or proceed with CPU if override turned on)
    try:
        gpu_in_use = int(os.environ['CUDA_VISIBLE_DEVICES'])
    except KeyError:
        gpu_in_use = 0
    gpu_info = get_gpu_information()[gpu_in_use]
    if not args.force and gpu_info['name'] == 'CPU':
        print('AlphaFold requires a GPU and none was detected - please re-run on a machine with a GPU.')
        sys.exit(2)

    # Check that there is enough free GPU RAM to run
    if not args.force and gpu_info['memory.free'] < 10000:
        print('AlphaFold requires a large amount of GPU ram available, and this machine has less than 10GB free. '
              'Most likely this means that another user is already using this GPU. Please try again on another '
              'machine. Available machines: https://nmrbox.org/hardware#details')
        sys.exit(3)

    # Only use GPU relaxation on the A100
    if args.gpu_relax is None:
        args.gpu_relax = gpu_info['name'] in ['A100-PCIE-40GB', 'NVIDIA A100-PCIE-40GB',
                                              'NVIDIA A100-SXM4-40GB', 'Tesla V100-PCIE-32GB']

    # Print run configuration
    if args.verbose:
        print(f"Detected GPU: {gpu_info['name']} ({gpu_info['memory.free']}MB free GPU RAM out of"
              f" {gpu_info['memory.total']}MB total)")
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

    # Build the command
    command = ['singularity', 'exec', '--nv', '-B', arguments.database, '-B',
               arguments.output, '-B', arguments.FASTA_file, args.singularity_container]

    if num_chains == 1:
        print('Found FASTA file with one sequence, treating as a monomer.')
        command.append('/opt/alphafold/monomer.sh')
    elif num_chains > 1:
        print(f'Found FASTA file with {num_chains} sequences, treating as a multimer.')
        command.append('/opt/alphafold/multimer.sh')
    command.extend([arguments.database, arguments.FASTA_file, arguments.output, arguments.max_template_date,
                    "--use_gpu_relax" if arguments.gpu_relax else "--nouse_gpu_relax"])

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
parser.add_argument("--database", "-d", action="store", default="/reboxitory/data/alphafold/2.2.0.2",
                    help='The path to the AlphaFold database to use for the calculation.')
parser.add_argument("--output-dir", "-o", action="store", default=".", dest='output',
                    help='The path where the output data should be stored. Defaults to the current directory.')
parser.add_argument("--max-template-date", "-t", action="store", default=str(datetime.date.today()),
                    dest='max_template_date',
                    help='If you are predicting the structure of a protein that is already in PDB'
                         ' and you wish to avoid using it as a template, then max-template-date must be set to'
                         ' be before the release date of the structure.')
parser.add_argument("--singularity-container", action="store",
                    default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'alphafold.sif'),
                    help=argparse.SUPPRESS)
# Allow force running on machines without GPU
parser.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                    help='Try to run even if no GPU is detected, or the memory is deemed insufficient. Not '
                         'recommended, as either failure or extremely long run times are expected.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Be verbose.")
# Allow testing with specific GPU relax setting
parser.add_argument('--gpu-relax', dest='gpu_relax', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--no-gpu-relax', dest='gpu_relax', action='store_false', help=argparse.SUPPRESS)
parser.set_defaults(gpu_relax=None)
parser.add_argument('FASTA_file', action="store",
                    help='The FASTA file to use for the calculation.')
args = parser.parse_args()

# Get absolute paths
args.database = abspath(args.database)
args.output = abspath(args.output)
args.FASTA_file = abspath(args.FASTA_file)
args.singularity_container = abspath(args.singularity_container)

run(args)
