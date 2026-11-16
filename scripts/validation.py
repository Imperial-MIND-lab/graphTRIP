"""
This script evaluates graphTRIP zero-shot on an independent dataset (psilodep1): no weight
is updated and no psilodep1 outcome is used.

Dependencies:
- experiments/configs/psilodep1_finetuning.json
- outputs/graphtrip/weights/                            graphTRIP weights

Outputs:
- outputs/validation/evaluate_graphtrip/                zero-shot graphTRIP

Author: Hanna M. Tolle
Date: 2025-12-07
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import copy
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


# Baseline severity scores that are harmonised onto the psilodep2 training scale. These are
# the only clinical inputs that vary within psilodep1: Condition and Stop_SSRI are constant
# there, so no affine map of them can change a correlation.
HARMONISE_ATTRS = ['QIDS_Before', 'BDI_Before']
N_PERMUTATIONS = 10000

# Dataset config keys that define what a model reads
INPUT_KEYS = ['graph_attrs', 'node_attrs', 'edge_attrs', 'context_attrs',
              'edge_tfm_type', 'edge_tfm_params', 'add_3Dcoords',
              'self_loop_fill_value', 'max_spd_dist', 'standardise_x']


def zeroshot_dataset_config(psilodep1_config, source_dataset):
    '''
    Dataset config for a zero-shot evaluation on psilodep1.

    The cohort-defining keys (study, session, atlas, target, batch size, folds) stay those of
    psilodep1; everything that defines the model's inputs is taken from the pretrained
    model's own dataset config, so that the model is transferred with the input domains it
    was trained on.
    '''
    dataset = copy.deepcopy(psilodep1_config['dataset'])
    for key in INPUT_KEYS:
        if key in source_dataset:
            dataset[key] = copy.deepcopy(source_dataset[key])
    dataset['graph_attrs_to_standardise'] = []
    return dataset


def source_standardised_attrs(source_dataset):
    '''
    Which clinical attributes the pretrained model received standardised.

    Returned as None when the saved config does not store a plain list, in which case the
    transfer experiment reads it from config.json.
    '''
    value = source_dataset.get('graph_attrs_to_standardise', None)
    return value if isinstance(value, list) else None


def zeroshot_config(psilodep1_config, source_config, weights_dir, ex_dir, seed, verbose,
                    n_permutations):
    '''
    Config updates for a zero-shot transfer of one pretrained model onto psilodep1, for the
    transfer_and_finetune experiment.
    '''
    source_dataset = source_config['dataset']
    config_updates = {}

    # Use the same model config as the pretrained model
    for ingredient in ['vgae_model', 'mlp_model']:
        if ingredient in source_config:
            config_updates[ingredient] = copy.deepcopy(source_config[ingredient])

    # But use the dataset config of psilodep1 (with the pretrained model's inputs)
    config_updates['dataset'] = zeroshot_dataset_config(psilodep1_config, source_dataset)

    # Experiment settings
    config_updates['num_epochs'] = 0  # no finetuning
    config_updates['harmonise_graph_attrs'] = [a for a in HARMONISE_ATTRS
                                               if a in source_dataset['graph_attrs']]
    config_updates['source_standardised_attrs'] = source_standardised_attrs(source_dataset)
    config_updates['n_permutations'] = n_permutations

    # Add output and weights directories
    config_updates['output_dir'] = ex_dir
    config_updates['weights_dir'] = weights_dir
    config_updates['save_weights'] = False
    config_updates['seed'] = seed
    config_updates['verbose'] = verbose
    return config_updates


def main(config_file, weights_base_dir, output_dir, verbose, debug, seed, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)
    weights_base_dir = add_project_root(weights_base_dir)

    # Make sure the config and weights base directory exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    if not os.path.exists(weights_base_dir):
        raise FileNotFoundError(f"{weights_base_dir} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)

    # Experiment settings
    observer = 'FileStorageObserver'
    n_permutations = 100 if debug else N_PERMUTATIONS

    # Weights directory is this seed's graphTRIP weights directory
    graphtrip_weights_dir = os.path.join(weights_base_dir, f'seed_{seed}')
    graphtrip_config = load_configs_from_json(os.path.join(graphtrip_weights_dir, 'config.json'))

    # Evaluate graphTRIP zero-shot on the validation dataset -----------------------
    exname = 'transfer_and_finetune'
    ex_dir = os.path.join(output_dir, 'evaluate_graphtrip', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = zeroshot_config(config, graphtrip_config, graphtrip_weights_dir,
                                         ex_dir, seed, verbose, n_permutations)
        run(exname, observer, config_updates)
    else:
        print(f"graphTRIP validation experiment already exists in {ex_dir}.")


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.validation
    -c experiments/configs/psilodep1_finetuning.json
    -w outputs/graphtrip/weights/
    -o outputs/validation/
    -s 0 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='experiments/configs/psilodep1_finetuning.json',
                        help='Path to the config file with psilodep1 validation model config')
    parser.add_argument('-w', '--weights_base_dir', type=str, default='outputs/graphtrip/weights/',
                        help='Path to the base directory with graphTRIP VGAE weights')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/validation/',
                        help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.weights_base_dir, args.output_dir,
         args.verbose, args.debug, args.seed, args.config_id)
