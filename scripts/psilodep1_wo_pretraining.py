"""
This scripts trains graphTRIP and a simple MLP model on the psilodep1 dataset.
No pretraining is done.

Dependencies of this script:
- experiments/configs/graphtrip.json

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
import copy
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(config_file, output_dir, verbose, debug, seed, jobid=0, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found.")
    
    # Load config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)

    # Determine random seed for this job
    seed = seed + jobid

    # Train model models sequentially
    model_names = ['control_mlp', 'graphtrip']
    for model_name in model_names:

        # Run the experiment if it doesn't exist
        ex_dir = os.path.join(output_dir, model_name, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = copy.deepcopy(config)

            # Set psilodep1 dataset configs ----------------------------------------
            if model_name == 'control_mlp':
                # Overwrite some dataset configs for MLP (only on clinical data)
                config_updates['dataset']['node_attrs'] = []
                config_updates['dataset']['edge_attrs'] = []
                config_updates['dataset']['edge_tfm_type'] = None
                config_updates['dataset']['edge_tfm_params'] = {}
                config_updates['dataset']['add_3Dcoords'] = False
                config_updates['dataset']['standardise_x'] = False

                # Delete VGAE and VGAE-training related configs
                del config_updates['vgae_model']
                del config_updates['alpha']
                del config_updates['num_z_samples']

            # Overwrite some dataset configs with Psilodep1 configs 
            config_updates['dataset']['study'] = 'psilodep1'
            config_updates['dataset']['session'] = 'before'
            config_updates['dataset']['target'] = 'QIDS_1week'
            config_updates['dataset']['batch_size'] = 7
            config_updates['dataset']['num_folds'] = 7
            config_updates['dataset']['context_attrs'] = [] # drug is constant in psilodep1

            # Add all graph attributes, available in psilodep1
            graph_attrs = ["HAMD_Before", 
                           "QIDS_Before", 
                           "LOTR_Before", 
                           "BDI_Before"]
            config_updates['dataset']['graph_attrs'] = graph_attrs
            config_updates['mlp_model']['extra_dim'] = len(graph_attrs)

            # Update VGAE model configs for graphTRIP
            if model_name == 'graphtrip':
                config_updates['vgae_model']['params']['num_graph_attr'] = len(graph_attrs)
                config_updates['vgae_model']['params']['num_context_attrs'] = 0
                
            # Run experiment --------------------------------------------------------
            # Experiment settings
            observer = 'FileStorageObserver'
            exname = 'train_jointly' if model_name == 'graphtrip' else 'train_mlp'    
            config_updates['output_dir'] = ex_dir
            config_updates['verbose'] = verbose
            config_updates['seed'] = seed
            config_updates['save_weights'] = False # don't need this
            if debug:
                config_updates['num_epochs'] = 2
            run(exname, observer, config_updates)
        else:
            print(f"{model_name} experiment already exists in {ex_dir}")

if __name__ == "__main__":
    """
    How to run:
    python psilodep1_wo_pretraining.py -c experiments/configs/graphtrip.json -o outputs/psilodep1/wo_pretraining/ -s 0 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/psilodep1/wo_pretraining', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Initial random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--jobid', type=int, default=0, help='Job ID')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()
        
    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.jobid, args.config_id)
