"""
This scripts validates the graphTRIP VGAE representations
on and independent dataset (psilodep1).

Dependencies:
- experiments/configs/psilodep1_finetuning.json

Outputs:
- outputs/validation/evaluate_graphtrip/
- outputs/validation/pretraining/
- outputs/validation/finetuning/

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

    # Evaluate graphTRIP on validation dataset ------------------------------------
    exname = 'transfer_and_finetune'
    ex_dir = os.path.join(output_dir, 'evaluate_graphtrip', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = {}
        # Use the same model config as graphTRIP
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])

        # But use the dataset config of psilodep1 (with graphTRIP graph_attrs)
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['dataset']['graph_attrs'] = graphtrip_config['dataset']['graph_attrs']

        # Experiment settings
        config_updates['num_epochs'] = 0 # no finetuning

        # Add output and weights directories
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = graphtrip_weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"graphTRIP validation experiment already exists in {ex_dir}.")

    # Pre-train new MLP on Psilodep2 --------------------------------------------
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
        incompatible_attrs = ['Condition', 'Stop_SSRI']
        new_graph_attrs = [attr for attr in graph_attrs if attr not in incompatible_attrs]
        config_updates['dataset']['graph_attrs'] = new_graph_attrs

        # Since we're not training end-to-end, we need to standardise clinical data
        config_updates['dataset']['graph_attrs_to_standardise'] = new_graph_attrs

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
        print(f"Pre-training MLP on Psilodep1 experiment already exists in {ex_dir}.")
    pretrain_dir = ex_dir

    # Transfer and finetune pre-trained MLP on Psilodep1 --------------------------------
    exname = 'transfer_and_finetune'
    ex_dir = os.path.join(output_dir, 'finetuning', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        # Use the psilodep1 finetuning config (training & dataset configs)
        config_updates = copy.deepcopy(config)

        # Add model configs from graphTRIP
        config_updates['vgae_model'] = copy.deepcopy(graphtrip_config['vgae_model'])
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])

        # Adapt clinical data dimensions
        graph_attrs = config_updates['dataset']['graph_attrs']
        config_updates['mlp_model']['extra_dim'] = len(graph_attrs)
        config_updates['vgae_model']['params']['num_graph_attr'] = len(graph_attrs)

        # Add output and weights directories
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = pretrain_dir
        config_updates['save_weights'] = True
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Finetuning pre-trained MLP on Psilodep1 experiment already exists in {ex_dir}.")

    # --------------------------------------------------------------------------
    # Validation Dataset Benchmarks 
    # --------------------------------------------------------------------------

    # 1. Linear regression trained on clinical data ----------------------------
    exname = 'train_linreg_on_clinical'
    ex_dir = os.path.join(output_dir, 'linreg_on_clinical_data', f'seed_{seed}')
    if not os.path.exists(add_project_root(ex_dir)):
        config_updates = {}

        # Dataset configs
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['dataset']['batch_size'] = -1 # linear regression uses full batch
        config_updates['dataset']['graph_attrs_to_standardise'] = ['QIDS_Before', 'BDI_Before']

        # Other configs
        config_updates['regression_model'] = 'LinearRegression'
        config_updates['output_dir'] = ex_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['save_weights'] = False 
        run(exname, observer, config_updates)
    else:
        print(f"Train linear regression on clinical data experiment already exists in {ex_dir}.")

    # 2. Control MLP trained on clinical data ----------------------------------
    exname = 'train_mlp'
    ex_dir = os.path.join(output_dir, 'control_mlp', f'seed_{seed}')
    if not os.path.exists(add_project_root(ex_dir)):
        # Use MLP model config from graphTRIP model and dataset config from psilodep1
        config_updates = {}
        config_updates['mlp_model'] = copy.deepcopy(graphtrip_config['mlp_model'])
        config_updates['dataset'] = copy.deepcopy(config['dataset'])

        # Training configs from graphTRIP
        config_updates['num_epochs'] = 2 if debug else graphtrip_config['num_epochs']
        config_updates['lr'] = graphtrip_config['lr']

        # Make sure we don't use any transforms on data that doesn't exist
        config_updates['dataset']['edge_tfm_type'] = None
        config_updates['dataset']['edge_tfm_params'] = {}
        config_updates['dataset']['add_3Dcoords'] = False
        config_updates['dataset']['standardise_x'] = False

        # Remove neuroimaging and context attributes
        config_updates['dataset']['node_attrs'] = []
        config_updates['dataset']['edge_attrs'] = []
        config_updates['dataset']['context_attrs'] = []

        # Add more demographic and clinical data available in psilodep1
        additional_attrs = ['HAMD_Before', 'LOTR_Before', 'Gender', 'Age']
        config_updates['dataset']['graph_attrs'] += additional_attrs
        numerical_attrs = ['QIDS_Before', 'BDI_Before', 'HAMD_Before', 'LOTR_Before', 'Age']
        config_updates['dataset']['graph_attrs_to_standardise'] = numerical_attrs

        # Adapt clinical data dimensions
        graph_attrs = config_updates['dataset']['graph_attrs']
        config_updates['mlp_model']['extra_dim'] = len(graph_attrs)

        # Saving, seed, and verbose configs
        config_updates['save_weights'] = False
        config_updates['output_dir'] = ex_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        
        # Run experiment
        run(exname, observer, config_updates)
    else:
        print(f"Control MLP benchmark experiment already exists in {ex_dir}.")

    # 3. Train a new graphTRIP model on psilodep1 --------------------------------
    exname = 'train_jointly'
    ex_dir = os.path.join(output_dir, 'psilodep1_graphtrip', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(graphtrip_config)
        config_updates['dataset'] = copy.deepcopy(config['dataset'])

        # Psilodep1-specific configs
        config_updates['balance_attrs'] = None # no k-fold split balancing based on Condition
        graph_attrs = config_updates['dataset']['graph_attrs']
        config_updates['mlp_model']['extra_dim'] = len(graph_attrs)

        # Debugging, saving, seed, and verbose configs
        config_updates['num_epochs'] = 2 if debug else graphtrip_config['num_epochs']
        config_updates['output_dir'] = ex_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Psilodep1 graphTRIP experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python validation.py -c experiments/configs/psilodep1_finetuning.json -o outputs/validation/ -s 0 -v -dbg -ci 0
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
    main(args.config_file, args.weights_base_dir, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
