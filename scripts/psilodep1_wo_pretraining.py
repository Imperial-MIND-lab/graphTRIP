"""
This scripts trains graphTRIP and a simple MLP model on the psilodep1 dataset.
No pretraining is done.

Dependencies of this script:
- experiments/configs/graphtrip.json
- experiments/configs/control_mlp.json

Author: Hanna M. Tolle
Date: 2025-05-31
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(model_name, config_file, output_dir, verbose, debug, seed, save_weights=False):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)
    output_dir = os.path.join(output_dir, model_name, f'seed_{seed}')

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found.")
    
    # Load config
    config = load_configs_from_json(config_file)
    config_updates = fetch_job_config(config, 0)
    
    # Set psilodep1 dataset configs ----------------------------------------
    # Initialise dataset configs for MLP
    if model_name == 'control_mlp':
        config_updates['dataset'] = {}
        config_updates['dataset']['node_attrs'] = []
        config_updates['dataset']['context_attrs'] = []
        config_updates['dataset']['graph_attrs'] = []
        config_updates['dataset']['edge_attrs'] = []
        config_updates['dataset']['edge_tfm_type'] = None
        config_updates['dataset']['edge_tfm_params'] = {}
        config_updates['dataset']['add_3Dcoords'] = False
        config_updates['dataset']['standardise_x'] = False

    # Overwrite some dataset configs with Psilodep1 configs 
    config_updates['dataset']['study'] = 'psilodep1'
    config_updates['dataset']['session'] = 'before'
    config_updates['dataset']['target'] = 'QIDS_1week'
    config_updates['dataset']['batch_size'] = 7
    config_updates['dataset']['num_folds'] = 7
    config_updates['dataset']['context_attrs'] = []

    # Add all graph attributes, available in psilodep1
    graph_attrs = ["Gender", 
                    "Age", 
                    "HAMD_Before", 
                    "QIDS_Before", 
                    "LOTR_Before", 
                    "BDI_Before"]
    config_updates['dataset']['graph_attrs'] = graph_attrs
    config_updates['mlp_model']['extra_dim'] = len(graph_attrs)

    # Update VGAE model configs for graphTRIP
    if model_name.lower() == 'graphtrip':
        config_updates['vgae_model']['params']['num_graph_attr'] = len(graph_attrs)
        config_updates['vgae_model']['params']['num_context_attrs'] = 0
        
    # Run experiment --------------------------------------------------------
    # Experiment settings
    observer = 'FileStorageObserver'
    exname = 'train_jointly' if model_name.lower() == 'graphtrip' else 'train_mlp'    

    # Run the experiment if it doesn't exist
    if not os.path.exists(output_dir):
        config_updates['output_dir'] = output_dir
        config_updates['verbose'] = verbose
        config_updates['seed'] = seed
        config_updates['run_name'] = f'{model_name}_psilodep1_seed_{seed}'
        config_updates['save_weights'] = save_weights
        if debug:
            config_updates['num_epochs'] = 2
        run(exname, observer, config_updates)
    else:
        print(f"{model_name} experiment already exists in {output_dir}")

if __name__ == "__main__":
    """
    How to run:
    python psilodep1_wo_pretraining.py -m graphtrip -j 0 -c experiments/configs/ -o outputs/psilodep1/wo_pretraining/ -s 291 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model to train. control_mlp or graphtrip')
    parser.add_argument('-j', '--job_id', type=int, required=True, help='Job ID')
    parser.add_argument('-c', '--config_dir', type=str, default='experiments/configs/', help='Path to the config directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/psilodep1/wo_pretraining', help='Path to the output directory')
    parser.add_argument('-s', '--init_seed', type=int, default=291, help='Initial random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Make sure debugging outputs don't overwrite any existing outputs
    if args.debug:
        if args.output_dir == 'outputs/psilodep1/wo_pretraining/':
            raise ValueError("output_dir must be specified when using debug mode.")
        
    # Get the config file and seed
    config_file = os.path.join(args.config_dir, f'{args.model_name.lower()}.json')
    seed = args.init_seed + args.job_id

    # Run the main function
    main(args.model_name, config_file, args.output_dir, args.verbose, args.debug, seed)
