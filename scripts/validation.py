"""
This script validates the graphTRIP VGAE representations
on an independent dataset (psilodep1).

Dependencies:
- experiments/configs/psilodep1_finetuning.json

Outputs:
- outputs/validation/evaluate_graphtrip/            zero-shot graphTRIP
- outputs/validation/permutation_importance/        importance of its MLP inputs
- outputs/validation/pretraining/                   MLPs re-trained on psilodep2
- outputs/validation/evaluate_pretrained/           those MLPs, zero-shot
- outputs/validation/permutation_importance_pretrained/

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
N_IMPORTANCE_REPEATS = 200

# Clinical attributes that do not carry information in psilodep1, and are therefore dropped
# when re-training an MLP for the zero-shot comparison
INCOMPATIBLE_ATTRS = ['Condition', 'Stop_SSRI']


def zeroshot_dataset_config(psilodep1_config, graph_attrs):
    '''
    Dataset config for a zero-shot evaluation on psilodep1.

    graph_attrs_to_standardise is cleared: the clinical inputs are mapped into each fold
    model's own input space explicitly by transfer_and_finetune, so leaving a value here
    would describe a transform that is never applied.
    '''
    dataset = copy.deepcopy(psilodep1_config['dataset'])
    dataset['graph_attrs'] = list(graph_attrs)
    dataset['graph_attrs_to_standardise'] = []
    return dataset


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
    if debug:
        config['num_epochs'] = 2

    # Weights directory is this seed's graphTRIP weights directory
    graphtrip_weights_dir = os.path.join(weights_base_dir, f'seed_{seed}')
    graphtrip_config = load_configs_from_json(os.path.join(graphtrip_weights_dir, 'config.json'))
    graphtrip_attrs = graphtrip_config['dataset']['graph_attrs']

    n_permutations = 100 if debug else N_PERMUTATIONS
    n_repeats = 5 if debug else N_IMPORTANCE_REPEATS

    # Evaluate graphTRIP zero-shot on the validation dataset -----------------------
    exname = 'transfer_and_finetune'
    ex_dir = os.path.join(output_dir, 'evaluate_graphtrip', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = {}
        # Use the same model config as graphTRIP
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])

        # But use the dataset config of psilodep1 (with graphTRIP graph_attrs)
        config_updates['dataset'] = zeroshot_dataset_config(config, graphtrip_attrs)

        # Experiment settings
        config_updates['num_epochs'] = 0  # no finetuning
        config_updates['harmonise_graph_attrs'] = HARMONISE_ATTRS
        config_updates['source_standardised_attrs'] = \
            graphtrip_config['dataset']['graph_attrs_to_standardise']
        config_updates['n_permutations'] = n_permutations

        # Add output and weights directories
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = graphtrip_weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"graphTRIP validation experiment already exists in {ex_dir}.")

    # How much of the zero-shot prediction is brain-derived -------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])
        config_updates['dataset'] = zeroshot_dataset_config(config, graphtrip_attrs)

        # Every fold model predicts every psilodep1 patient, so features are shuffled across
        # the whole cohort rather than within test folds
        config_updates['mode'] = 'mean_vote'
        config_updates['harmonise_graph_attrs'] = HARMONISE_ATTRS
        config_updates['source_standardised_attrs'] = \
            graphtrip_config['dataset']['graph_attrs_to_standardise']
        config_updates['n_repeats'] = n_repeats

        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = graphtrip_weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Permutation importance on Psilodep1 experiment already exists in {ex_dir}.")

    # Pre-train new MLP on Psilodep2 --------------------------------------------
    # Same frozen graphTRIP encoder, but only the clinical features that carry information
    # in psilodep1, so the head can use them fully.
    exname = 'retrain_mlp'
    ex_dir = os.path.join(output_dir, 'pretraining', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        # Set the VGAE and MLP model and dataset config as graphTRIP
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])
        config_updates['dataset'] = copy.deepcopy(graphtrip_config['dataset'])

        # Remove graph_attrs that are not compatible with psilodep1
        graph_attrs = config_updates['dataset']['graph_attrs']
        new_graph_attrs = [attr for attr in graph_attrs if attr not in INCOMPATIBLE_ATTRS]
        config_updates['dataset']['graph_attrs'] = new_graph_attrs

        # Since we're not training end-to-end, we need to standardise clinical data.
        # This must be a copy: sacred serialises config.json with jsonpickle, and two config
        # keys sharing one list object are written as a back-reference that cannot be
        # resolved from the saved file.
        config_updates['dataset']['graph_attrs_to_standardise'] = list(new_graph_attrs)

        # Adapt clinical data dimensions
        config_updates['mlp_model']['extra_dim'] = len(new_graph_attrs)
        config_updates['vgae_model']['params']['num_graph_attr'] = len(new_graph_attrs)

        # Training configs
        config_updates['num_epochs'] = 2 if debug else graphtrip_config['num_epochs']
        config_updates['mlp_lr'] = graphtrip_config['lr']
        config_updates['num_z_samples'] = graphtrip_config['num_z_samples']
        config_updates['alpha'] = 0              # VGAE is frozen
        config_updates['vgae_lr'] = 0.0          # VGAE is frozen
        config_updates['reinit_pooling'] = False # Re-use trained, frozen pooling module

        # Add output and weights directories
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = graphtrip_weights_dir
        config_updates['save_weights'] = True
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Pre-training MLP on Psilodep2 experiment already exists in {ex_dir}.")
    pretrain_dir = ex_dir

    # The pretrained MLPs were trained on standardised clinical inputs. 
    pretrained_attrs = [a for a in graphtrip_attrs if a not in INCOMPATIBLE_ATTRS]

    # Evaluate the pre-trained MLPs zero-shot on Psilodep1 -------------------------
    exname = 'transfer_and_finetune'
    ex_dir = os.path.join(output_dir, 'evaluate_pretrained', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])
        config_updates['dataset'] = zeroshot_dataset_config(config, pretrained_attrs)

        # Adapt clinical data dimensions
        config_updates['mlp_model']['extra_dim'] = len(pretrained_attrs)
        config_updates['vgae_model']['params']['num_graph_attr'] = len(pretrained_attrs)

        config_updates['num_epochs'] = 0  # no finetuning
        config_updates['harmonise_graph_attrs'] = HARMONISE_ATTRS
        config_updates['source_standardised_attrs'] = pretrained_attrs
        config_updates['n_permutations'] = n_permutations

        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = pretrain_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Zero-shot evaluation of pre-trained MLPs already exists in {ex_dir}.")

    # How much of that prediction is brain-derived ---------------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance_pretrained', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])
        config_updates['dataset'] = zeroshot_dataset_config(config, pretrained_attrs)
        config_updates['mlp_model']['extra_dim'] = len(pretrained_attrs)
        config_updates['vgae_model']['params']['num_graph_attr'] = len(pretrained_attrs)

        config_updates['mode'] = 'mean_vote'
        config_updates['harmonise_graph_attrs'] = HARMONISE_ATTRS
        config_updates['source_standardised_attrs'] = pretrained_attrs
        config_updates['n_repeats'] = n_repeats

        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = pretrain_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Permutation importance of pre-trained MLPs already exists in {ex_dir}.")


if __name__ == "__main__":
    """
    How to run:
    python validation.py
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
