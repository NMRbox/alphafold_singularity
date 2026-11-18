#!/usr/bin/env python3.12

import argparse
import configparser
import csv
import dataclasses
import json
import os
import string
import subprocess
import sys
from io import StringIO
from os.path import abspath
from typing import List, Optional, Tuple, Union

# Production layout: this script lives in the AlphaFold 3 database directory, next to the container and the model
#  weights. Any of these can be overridden by a config file (for testing) or on the command line.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'alphafold3.ini')
DEFAULT_PATHS = {
    'singularity_container': os.path.join(SCRIPT_DIR, 'alphafold3.sif'),
    'database': SCRIPT_DIR,
    'models': os.path.join(SCRIPT_DIR, 'models'),
}

# XLA flag set by default in the AlphaFold 3 Dockerfile - it must be retained if we override XLA_FLAGS
DEFAULT_XLA_FLAGS = '--xla_gpu_enable_triton_gemm=false'
# Required by AlphaFold 3 on compute capability 7.x GPUs (e.g. V100)
CC7_XLA_FLAGS = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'


@dataclasses.dataclass
class GPUInfo:
    name: str = 'CPU'
    utilization_gpu: Union[int, float] = float('nan')
    memory_free: Union[int, float] = float('nan')
    memory_total: Union[int, float] = float('nan')
    index: Union[int, float] = None
    uuid: str = 'Unknown'
    compute_capability: Optional[float] = None


def query_nvidia_smi(fields: List[str]) -> List[List[str]]:
    result = subprocess.run(['/usr/bin/nvidia-smi', f'--query-gpu={",".join(fields)}', '--format=csv,nounits,noheader'],
                            check=True, capture_output=True)
    return [[_.strip() for _ in row] for row in csv.reader(StringIO(result.stdout.decode())) if row]


def get_gpu_information() -> GPUInfo:
    fields = ['name', 'utilization.gpu', 'memory.free', 'memory.total', 'index', 'uuid']
    try:
        try:
            csv_data = query_nvidia_smi(fields + ['compute_cap'])
        except subprocess.CalledProcessError:
            # Older drivers don't support querying the compute capability
            csv_data = query_nvidia_smi(fields)
        available_gpus = [GPUInfo(name=_[0],
                                  utilization_gpu=int(_[1]),
                                  memory_free=int(_[2]),
                                  memory_total=int(_[3]),
                                  index=int(_[4]),
                                  uuid=_[5],
                                  compute_capability=float(_[6]) if len(_) > 6 else None) for _ in csv_data]
        if len(available_gpus) == 0:
            available_gpus.append(GPUInfo())

        # Check that they have a GPU (or proceed with CPU if override turned on)
        try:
            gpu_in_use = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0].strip()
            if not gpu_in_use:
                # An empty CUDA_VISIBLE_DEVICES hides all GPUs
                return GPUInfo()
            try:
                gpu_in_use = int(gpu_in_use)
                for gpu in available_gpus:
                    if gpu.index == gpu_in_use:
                        return gpu
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
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    yield name, ''.join(seq)
                name, seq = line, []
            else:
                seq.append(line)
        if name:
            yield name, ''.join(seq)


def chain_ids():
    """ Generates A, B, ..., Z, AA, AB, ... as AlphaFold 3 chain IDs. """
    letters = string.ascii_uppercase
    length = 1
    while True:
        for i in range(len(letters) ** length):
            chain_id = ''
            for _ in range(length):
                i, remainder = divmod(i, len(letters))
                chain_id = letters[remainder] + chain_id
            yield chain_id
        length += 1


def fasta_to_alphafold3_json(fasta_file: str, num_seeds: int) -> Tuple[dict, int]:
    """ Converts a FASTA file of protein sequences into an AlphaFold 3 input dictionary. Identical sequences are
    combined into a single entity with multiple chain IDs, as AlphaFold 3 expects for homomers.

    Returns the AlphaFold 3 input dictionary and the number of chains. """

    sequences = list(read_fasta(fasta_file))
    if len(sequences) == 0:
        raise ValueError('Your FASTA file doesn\'t appear to be valid. Please consult documentation here: '
                         'https://en.wikipedia.org/wiki/FASTA_format')
    for _, sequence in sequences:
        if not sequence:
            raise ValueError('Your FASTA file contains an entry with no sequence.')
        if sequence.upper() != sequence:
            raise ValueError('Your FASTA file doesn\'t appear to be valid. All residues must be specified '
                             f'using capital letters. Problem in sequence: {sequence}')
        if not sequence.isalpha():
            raise ValueError('Your FASTA file doesn\'t appear to be valid. Sequences may only contain one-letter '
                             f'residue codes. Problem in sequence: {sequence}')

    id_generator = chain_ids()
    entities = {}
    for _, sequence in sequences:
        entities.setdefault(sequence, []).append(next(id_generator))

    job_name = os.path.splitext(os.path.basename(fasta_file))[0]
    af3_input = {
        'name': job_name,
        'modelSeeds': list(range(1, num_seeds + 1)),
        'sequences': [{'protein': {'id': ids if len(ids) > 1 else ids[0], 'sequence': sequence}}
                      for sequence, ids in entities.items()],
        'dialect': 'alphafold3',
        'version': 1,
    }
    return af3_input, len(sequences)


def estimate_tokens(af3_input: Union[dict, list]) -> int:
    """ Estimates the number of tokens for an AlphaFold 3 input. Polymers are one token per residue; ligands are not
    counted (they are one token per heavy atom, which we can't determine without the CCD). """

    def count_ids(chain_id):
        return len(chain_id) if isinstance(chain_id, list) else 1

    tokens = 0
    # AlphaFold Server dialect: a list of jobs
    if isinstance(af3_input, list):
        for job in af3_input:
            for entity in job.get('sequences', []):
                for key in ('proteinChain', 'rnaSequence', 'dnaSequence'):
                    if key in entity:
                        tokens += len(entity[key].get('sequence', '')) * entity[key].get('count', 1)
        return tokens

    for entity in af3_input.get('sequences', []):
        for key in ('protein', 'rna', 'dna'):
            if key in entity:
                tokens += len(entity[key].get('sequence', '')) * count_ids(entity[key].get('id', 'A'))
    return tokens


def load_alphafold3_json(json_file: str) -> Union[dict, list]:
    try:
        with open(json_file, 'r') as fp:
            af3_input = json.load(fp)
    except json.JSONDecodeError as err:
        raise ValueError(f'Your input JSON file \'{json_file}\' is not valid JSON: {err}') from None

    if isinstance(af3_input, list):
        # AlphaFold Server dialect
        return af3_input
    if not isinstance(af3_input, dict):
        raise ValueError(f'Your input JSON file \'{json_file}\' doesn\'t appear to be an AlphaFold 3 input file.')
    for required in ('name', 'sequences', 'modelSeeds'):
        if required not in af3_input:
            raise ValueError(f'Your input JSON file \'{json_file}\' is missing the required \'{required}\' field. '
                             'Please consult the AlphaFold 3 input documentation here: '
                             'https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md')
    if not af3_input['sequences']:
        raise ValueError(f'Your input JSON file \'{json_file}\' does not specify any sequences.')
    return af3_input


def max_supported_tokens(gpu_info: GPUInfo) -> int:
    """ Approximate maximum input sizes from the AlphaFold 3 performance documentation:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md """
    if gpu_info.memory_total >= 70000:
        return 5120
    if gpu_info.memory_total >= 35000:
        return 4352
    if gpu_info.compute_capability is not None and gpu_info.compute_capability < 8.0:
        return 1280 if gpu_info.compute_capability >= 7.0 else 1024
    return 1024


def resolve_paths(arguments):
    """ Determine container, database, and model paths. Priority: command line, then config file, then the
    production defaults. """

    config = configparser.ConfigParser()
    if arguments.config:
        if not os.path.isfile(arguments.config):
            raise IOError(f"The specified config file '{arguments.config}' does not exist.")
        config.read(arguments.config)
    elif os.path.isfile(DEFAULT_CONFIG_FILE):
        arguments.config = DEFAULT_CONFIG_FILE
        config.read(DEFAULT_CONFIG_FILE)
    config_section = config['alphafold3'] if config.has_section('alphafold3') else {}

    for key, default in DEFAULT_PATHS.items():
        if getattr(arguments, key) is None:
            setattr(arguments, key, config_section.get(key, default))
        setattr(arguments, key, abspath(os.path.expanduser(getattr(arguments, key))))

    if not os.path.isfile(arguments.singularity_container):
        raise IOError(f"The AlphaFold 3 container '{arguments.singularity_container}' does not exist.")
    if not os.path.isdir(arguments.database):
        raise IOError(f"The AlphaFold 3 database directory '{arguments.database}' does not exist.")
    if not os.path.isdir(arguments.models):
        raise IOError(f"The AlphaFold 3 model parameter directory '{arguments.models}' does not exist.")


def run(arguments):
    resolve_paths(arguments)

    # Ensure output directory exists and is writeable
    try:
        os.makedirs(arguments.output, exist_ok=True)
        test_path = os.path.join(arguments.output, '.KD5cpxqYzBNqaBZ66guDuh33ns7JYz2jrKq')
        with open(test_path, 'w'):
            pass
        os.unlink(test_path)
    except (IOError, PermissionError):
        raise IOError(f"Your specified output directory '{arguments.output}' is not "
                      f"writeable. Please choose a different output directory.") from None

    gpu_info = get_gpu_information()
    if not arguments.force and gpu_info.name == 'CPU':
        print('AlphaFold 3 requires a GPU and none was detected - please re-run on a machine with a GPU.')
        sys.exit(2)

    # Check that there is enough free GPU RAM to run
    if not arguments.force and gpu_info.memory_free < 10000:
        print('AlphaFold 3 requires a large amount of GPU ram available, and this machine has less than 10GB free. '
              'Most likely this means that another user is already using this GPU. Please try again on another '
              'machine. Available machines: https://nmrbox.org/hardware#details')
        sys.exit(3)

    if not arguments.force and gpu_info.compute_capability is not None and gpu_info.compute_capability < 6.0:
        print(f'AlphaFold 3 requires a GPU with compute capability 6.0 or newer, and this machine\'s GPU '
              f'({gpu_info.name}) has compute capability {gpu_info.compute_capability}. Please try again on another '
              'machine. Available machines: https://nmrbox.org/hardware#details')
        sys.exit(4)

    # Use unified memory (allowing spill-over into system RAM) on GPUs with less than 80GB of memory
    if arguments.unified_memory is None:
        arguments.unified_memory = not gpu_info.memory_total >= 70000

    # The Triton and cuDNN flash attention implementations require an Ampere (8.0) or newer GPU
    if arguments.flash_attention_implementation is None:
        if gpu_info.compute_capability is not None and gpu_info.compute_capability < 8.0:
            arguments.flash_attention_implementation = 'xla'
        else:
            arguments.flash_attention_implementation = 'triton'

    # Print run configuration
    if arguments.verbose:
        print(f"Detected GPU: {gpu_info.name} ({gpu_info.memory_free}MB free GPU RAM out of"
              f" {gpu_info.memory_total}MB total, compute capability {gpu_info.compute_capability})")
        print(f"Unified memory: {arguments.unified_memory}")
        print(f"Flash attention implementation: {arguments.flash_attention_implementation}")
        print(f"Config file: {arguments.config}")
        print(f"Container: {arguments.singularity_container}")
        print(f"Database: {arguments.database}")
        print(f"Model parameters: {arguments.models}")

    # Build the basics of the command
    command = ['singularity', 'exec', '--nv',
               '-B', f'{arguments.database}:{arguments.database}:ro',
               '-B', f'{arguments.models}:{arguments.models}:ro',
               '-B', arguments.output]

    # Determine the input - a FASTA file (converted to AlphaFold 3 JSON), an AlphaFold 3 JSON file, or a directory
    #  of AlphaFold 3 JSON files
    if os.path.isdir(arguments.input):
        print('Input is a directory - all AlphaFold 3 JSON files within will be processed.')
        input_arguments = ['--input_dir', arguments.input]
        command.extend(['-B', f'{arguments.input}:{arguments.input}:ro'])
        total_tokens = None
    elif arguments.input.lower().endswith('.json'):
        af3_input = load_alphafold3_json(arguments.input)
        total_tokens = estimate_tokens(af3_input)
        input_arguments = ['--json_path', arguments.input]
        # Bind the containing directory, so that any MSA or template files referenced by relative path are available
        input_dir = os.path.dirname(arguments.input)
        command.extend(['-B', f'{input_dir}:{input_dir}:ro'])
    else:
        af3_input, num_chains = fasta_to_alphafold3_json(arguments.input, arguments.num_seeds)
        total_tokens = estimate_tokens(af3_input)
        if num_chains == 1:
            print('Found FASTA file with one sequence, treating as a monomer.')
        else:
            print(f'Found FASTA file with {num_chains} sequences, treating as a multimer.')
        json_path = os.path.join(arguments.output, f"{af3_input['name']}_alphafold3_input.json")
        with open(json_path, 'w') as fp:
            json.dump(af3_input, fp, indent=2)
        print(f'Wrote AlphaFold 3 input file: {json_path}. You can edit this file and provide it as input to '
              f'customize the run (e.g. to add ligands, DNA, RNA, or modifications).')
        input_arguments = ['--json_path', json_path]

    if total_tokens and total_tokens > max_supported_tokens(gpu_info):
        print(f'AlphaFold 3 uses memory in accordance with the number of tokens (approximately one per residue, '
              f'plus one per ligand atom) in all chains. Your input has approximately {total_tokens} tokens, which '
              f'is more than the {max_supported_tokens(gpu_info)} tokens that are known to work on this GPU '
              f'({gpu_info.name}), so you may run out of memory. Please try rerunning on a machine with an 80GB '
              f'A100 or H100 GPU. For machine details, please see: https://nmrbox.org/hardware#details')

    # Map the system cuda into the container
    if arguments.cuda_dir:
        if not os.path.isdir(arguments.cuda_dir):
            raise IOError('The specified CUDA library directory does not exist.')
        command.extend(['-B', f'{arguments.cuda_dir}:/usr/local/cuda'])

    # Environment variables must be set with --env, as those set in the container (from the AlphaFold 3 Dockerfile)
    #  otherwise take precedence over the host environment
    xla_flags = DEFAULT_XLA_FLAGS
    if gpu_info.compute_capability is not None and 7.0 <= gpu_info.compute_capability < 8.0:
        xla_flags = CC7_XLA_FLAGS
    command.extend(['--env', f'XLA_FLAGS={xla_flags}'])
    if arguments.unified_memory:
        command.extend(['--env', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
                        '--env', 'TF_FORCE_UNIFIED_MEMORY=true',
                        '--env', 'XLA_CLIENT_MEM_FRACTION=3.2'])

    # Finish the singularity arguments. Beyond this line are the pass-through AlphaFold 3 arguments.
    command.extend([arguments.singularity_container, 'python', '/app/alphafold/run_alphafold.py'])

    # Limit the MSA tools to the CPUs this process is allowed to use (AlphaFold 3 would otherwise use up to 8,
    #  regardless of any CPU restrictions)
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count() or 1
    num_cpus = str(min(num_cpus, 8))

    command.extend(input_arguments)
    command.extend([
        '--output_dir', arguments.output,
        '--db_dir', arguments.database,
        '--model_dir', arguments.models,
        '--flash_attention_implementation', arguments.flash_attention_implementation,
        '--jackhmmer_n_cpu', num_cpus,
        '--nhmmer_n_cpu', num_cpus,
        '--run_data_pipeline' if arguments.run_data_pipeline else '--norun_data_pipeline',
        '--run_inference' if arguments.run_inference else '--norun_inference',
    ])

    # Optional pass-through arguments, only provided if specified, as they are not supported by all AlphaFold 3
    #  versions
    for option in ('max_template_date', 'num_recycles', 'num_diffusion_samples', 'buckets'):
        value = getattr(arguments, option)
        if value is not None:
            command.extend([f'--{option}', str(value)])
    if arguments.save_embeddings:
        command.append('--save_embeddings')

    print(f'Running AlphaFold 3, this will take a long time. All output following this line is from AlphaFold 3:')
    if arguments.verbose:
        print(f'Executing command: {" ".join(command)}')

    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp('singularity', command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AlphaFold 3. The input may be a FASTA file of protein '
                                                 'sequences, an AlphaFold 3 input JSON file, or a directory of '
                                                 'AlphaFold 3 input JSON files.')
    parser.add_argument("--output-dir", "-o", action="store", default=".", dest='output',
                        help='The path where the output data should be stored. Defaults to the current directory.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Be verbose.")
    parser.add_argument('--version', action='version', version='%(prog)s 3.0.0')
    parser.add_argument('input', action="store", metavar='INPUT',
                        help='The FASTA file, AlphaFold 3 JSON file, or directory of AlphaFold 3 JSON files to use for '
                             'the calculation. JSON files must end in ".json".')
    parser.add_argument('--num-seeds', dest='num_seeds', action='store', type=int, default=1,
                        help='When the input is a FASTA file, the number of model seeds to use. Each seed generates '
                             'a set of diffusion samples (5 by default). Ignored for JSON input, which specifies its '
                             'own model seeds. Defaults to 1.')

    advanced = parser.add_argument_group('advanced options')
    advanced.add_argument('--config', action='store', default=None,
                          help=f'A config file specifying the container, database, and model paths in an '
                               f'[alphafold3] section. Defaults to {DEFAULT_CONFIG_FILE} if it exists.')
    advanced.add_argument("--database", "-d", action="store", default=None,
                          help='The path to the AlphaFold 3 database to use for the calculation.')
    advanced.add_argument("--models", "-m", action="store", default=None,
                          help='The path to the directory containing the AlphaFold 3 model parameters.')
    advanced.add_argument("--cuda-dir", dest='cuda_dir', action="store", default=None,
                          help='The path to the CUDA library to use.')
    advanced.add_argument("--singularity-container", action="store", default=None, help=argparse.SUPPRESS)
    advanced.add_argument('--unified-memory', dest='unified_memory', action='store_true', help=argparse.SUPPRESS)
    advanced.add_argument('--no-unified-memory', dest='unified_memory', action='store_false', help=argparse.SUPPRESS)
    advanced.set_defaults(unified_memory=None)
    advanced.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                          help='Try to run even if no GPU is detected, or the memory is deemed insufficient. Not '
                               'recommended, as either failure or extremely long run times are expected.')

    alphafold_group = parser.add_argument_group('advanced AlphaFold 3 pass-through options')
    alphafold_group.add_argument("--max_template_date", action="store", default=None,
                                 help='If you are predicting the structure of a protein that is already in PDB'
                                      ' and you wish to avoid using it as a template, then max_template_date must be'
                                      ' set to be before the release date of the structure. Format: YYYY-MM-DD. '
                                      'AlphaFold 3 defaults to 2021-09-30.')
    alphafold_group.add_argument("--no_data_pipeline", dest='run_data_pipeline', action="store_false",
                                 help='Skip the data pipeline (MSA and template search). Use this if your input JSON '
                                      'already contains MSAs and templates, such as the *_data.json file written by a '
                                      'previous AlphaFold 3 run.')
    alphafold_group.add_argument("--no_inference", dest='run_inference', action="store_false",
                                 help='Only run the data pipeline (MSA and template search), not the structure '
                                      'prediction. The resulting *_data.json file can be used as input to a later run '
                                      'with --no_data_pipeline, which can be run on a different machine.')
    alphafold_group.add_argument("--flash_attention_implementation", action="store", default=None,
                                 choices=['triton', 'cudnn', 'xla'],
                                 help='Flash attention implementation to use. Defaults to triton on Ampere or newer '
                                      'GPUs, and xla otherwise.')
    alphafold_group.add_argument("--num_recycles", action="store", type=int, default=None,
                                 help='Number of recycles to use during inference. AlphaFold 3 defaults to 10.')
    alphafold_group.add_argument("--num_diffusion_samples", action="store", type=int, default=None,
                                 help='Number of diffusion samples to generate per seed. AlphaFold 3 defaults to 5.')
    alphafold_group.add_argument("--buckets", action="store", default=None,
                                 help='Comma-separated list of strictly increasing token sizes for the compilation '
                                      'buckets. Useful to avoid recompilation for inputs of similar sizes.')
    alphafold_group.add_argument("--save_embeddings", action="store_true", default=False,
                                 help='Save the final trunk single and pair embeddings in the output.')

    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error('--num-seeds must be at least 1.')
    if not args.run_data_pipeline and not args.run_inference:
        parser.error('--no_data_pipeline and --no_inference cannot both be specified, as there would be nothing to do.')

    # Get absolute paths (the container, database, and model paths are resolved later, as they may come from config)
    args.output = abspath(args.output)
    args.input = abspath(args.input)
    if not os.path.exists(args.input):
        parser.error(f"The input '{args.input}' does not exist.")
    if args.config:
        args.config = abspath(args.config)

    run(args)
