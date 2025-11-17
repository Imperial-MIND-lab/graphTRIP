"""
This scripts trains three benchmark models:
- a control MLP
- a PCA-reduced MLP
- a t-SNE-reduced MLP

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/benchmarks/control_mlp/
- outputs/benchmarks/pca_mlp/
- outputs/benchmarks/tsne_mlp/

Author: Hanna M. Tolle
Date: 2025-05-31
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


def main(config_file, output_dir, verbose, debug, seed, jobid=-1, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    
    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    # 1. Control MLP benchmark -----------------------------------------------
    if jobid == 0 or jobid == -1:
        exname = 'train_mlp'
        ex_dir = os.path.join(output_dir, 'control_mlp', f'seed_{seed}')
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same ingredient configs as the main model
            config_updates = copy.deepcopy(config)
            
            # Remove vgae_model and irrelevant training configs
            del config_updates['vgae_model']
            del config_updates['num_z_samples']
            del config_updates['alpha']

            # Make sure we don't use any transforms on data that doesn't exist
            config_updates['dataset']['edge_tfm_type'] = None
            config_updates['dataset']['edge_tfm_params'] = {}
            config_updates['dataset']['add_3Dcoords'] = False
            config_updates['dataset']['standardise_x'] = False

            # Set neuroimaging and context attributes to empty lists
            config_updates['dataset']['node_attrs'] = []
            config_updates['dataset']['edge_attrs'] = []
            config_updates['dataset']['context_attrs'] = []

            # Set MLP hidden_dim to latent_dim of graphTRIP model 
            # (graphTRIP has hidden_dim = None, which will set it to latent_dim)
            graphTRIP_latent_dim = config['vgae_model']['params']['latent_dim']
            config_updates['mlp_model']['params']['hidden_dim'] = graphTRIP_latent_dim

            # Add more config to the config_updates
            config_updates['save_weights'] = False
            config_updates['output_dir'] = ex_dir       
            
            # Run experiment
            run(exname, observer, config_updates)
            
    # 2. PCA benchmark -------------------------------------------------------
    if jobid == 1 or jobid == -1:
        exname = 'pca_benchmark'
        ex_dir = os.path.join(output_dir, 'pca_benchmark', f'seed_{seed}')

        # Check if the experiment has already been run
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same MLP and dataset configs as the main model
            config_updates = {}
            config_updates['mlp_model'] = copy.deepcopy(config['mlp_model'])
            config_updates['dataset'] = copy.deepcopy(config['dataset'])

            # Make sure the dataset has no edge transform
            config_updates['dataset']['edge_tfm_type'] = None
            config_updates['dataset']['edge_tfm_params'] = {}

            # Add PCA and training configs
            config_updates['n_components'] = 32
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            
            # Add more config to the config_updates
            config_updates['save_weights'] = False
            config_updates['output_dir'] = ex_dir
            
            # Run experiment
            run(exname, observer, config_updates)

    # 3. t-SNE benchmark -----------------------------------------------------
    if jobid == 2 or jobid == -1:
        exname = 'tsne_benchmark'
        ex_dir = os.path.join(output_dir, 'tsne_benchmark', f'seed_{seed}')

        # Check if the experiment has already been run
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same MLP and dataset configs as the main model
            config_updates = {}
            config_updates['mlp_model'] = copy.deepcopy(config['mlp_model'])
            config_updates['dataset'] = copy.deepcopy(config['dataset'])

            # Make sure the dataset has no edge transform
            config_updates['dataset']['edge_tfm_type'] = None
            config_updates['dataset']['edge_tfm_params'] = {}

            # Add PCA and training configs
            config_updates['n_components'] = 3 # that's the max for sklearn t-SNE
            config_updates['perplexity'] = 30
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            
            # Add more config to the config_updates
            config_updates['save_weights'] = False
            config_updates['output_dir'] = ex_dir
            
            # Run experiment
            run(exname, observer, config_updates)

if __name__ == "__main__":
    """
    How to run:
    python benchmarks.py -c experiments/configs/graphtrip.json -o outputs/benchmarks/ -s 291 -v -dbg -j 0 -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/benchmarks/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--jobid', type=int, default=-1, help='Job ID. If -1, runs all jobs sequentially.')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.jobid, args.config_id)
