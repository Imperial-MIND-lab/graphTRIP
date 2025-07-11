"""
This scripts trains the main graphTRIP model on the primary dataset,
and performs all post-hoc analyses.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/graphtrip/weights/
- outputs/graphtrip/permutation_importance/
- outputs/graphtrip/transfer_atlas/schaefer200/
- outputs/graphtrip/transfer_atlas/aal/
- outputs/graphtrip/attention_weights/
- outputs/graphtrip/grail/
- outputs/graphtrip/grail_posthoc/
- outputs/graphtrip/test_biomarkers/

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
import json
import pandas as pd
import numpy as np
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from utils.statsalg import grail_posthoc_analysis
from experiments.run_experiment import run


def main(config_file, output_dir, verbose, debug, seed, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    
    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
    ingredient_config = {
        'dataset': copy.deepcopy(config['dataset']),
        'vgae_model': copy.deepcopy(config['vgae_model']),
        'mlp_model': copy.deepcopy(config['mlp_model'])
    }
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2
    ingredient_config['seed'] = seed
    ingredient_config['verbose'] = verbose

    # Train graphTRIP ----------------------------------------------------------
    exname = 'train_jointly'
    weights_dir = os.path.join(output_dir, 'weights')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        run(exname, observer, config_updates)
    else:
        print(f"graphTRIP experiment already exists in {weights_dir}.")

    # Permutation importance ---------------------------------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['n_repeats'] = 50 if not debug else 2
        run(exname, observer, config_updates)
    else:
        print(f"Permutation importance experiment already exists in {ex_dir}.")

    # Transfer to Schaefer200 and AAL -----------------------------------------
    exname = 'test_and_finetune'
    atlases = ['schaefer200', 'aal']

    for atlas in atlases:
        ex_dir = os.path.join(output_dir, 'transfer_atlas', atlas)
        if not os.path.exists(ex_dir):
            # Get ingredient configs
            config_updates = copy.deepcopy(ingredient_config)
            
            # Change the atlas
            config_updates['dataset']['atlas'] = atlas
            num_nodes = 200 if atlas == 'schaefer200' else 116
            config_updates['dataset']['num_nodes'] = num_nodes
            config_updates['vgae_model']['params']['num_nodes'] = num_nodes

            # Other settings
            config_updates['num_epochs'] = 0 # no finetuning
            config_updates['output_dir'] = ex_dir
            config_updates['weights_dir'] = weights_dir

            run(exname, observer, config_updates)
        else:
            print(f"Transfer to {atlas} experiment already exists in {ex_dir}.")

    # Attention weights -------------------------------------------------------
    exname = 'attention_weights'
    ex_dir = os.path.join(output_dir, 'attention_weights')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Attention weights experiment already exists in {ex_dir}.")

    # GRAIL -------------------------------------------------------------------
    exname = 'grail'
    ex_dir = os.path.join(output_dir, 'grail')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['num_z_samples'] = 100 if not debug else 2
        config_updates['sigma'] = 2.0
        config_updates['all_rsn_conns'] = False
        run(exname, observer, config_updates)
    else:
        print(f"GRAIL experiment already exists in {ex_dir}.")

    # GRAIL posthoc ----------------------------------------------------------
    exname = 'grail_posthoc'
    ex_dir = os.path.join(output_dir, 'grail_posthoc')
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir, exist_ok=True)
        grail_dir = os.path.join(output_dir, 'grail')

        # Compute fold-mean alignments
        fold_dfs = []
        for k in range(config['dataset']['num_folds']):
            fold_df = pd.read_csv(os.path.join(grail_dir, f'k{k}_mean_alignments.csv'))
            fold_dfs.append(fold_df)
        mean_alignments = pd.concat(fold_dfs).groupby(level=0).mean()
        mean_alignments.to_csv(os.path.join(ex_dir, 'mean_alignments.csv'))

        # Analysis settings
        filter_percentile = 75
        num_seeds = 10
        seed = 291

        # Save analysis config
        grail_posthoc_config = {
            'filter_percentile': filter_percentile,
            'num_seeds': num_seeds,
            'seed': seed,
        }
        with open(os.path.join(ex_dir, 'config.json'), 'w') as f:
            json.dump(grail_posthoc_config, f, indent=4)

        # Run analysis
        cluster_labels, features_filtered = grail_posthoc_analysis(mean_alignments,
                                                                   num_seeds, seed,
                                                                   filter_percentile)
        # Save results
        np.savetxt(os.path.join(ex_dir, 'cluster_labels.csv'), cluster_labels, delimiter=',')
        with open(os.path.join(ex_dir, 'features_filtered.json'), 'w') as f:
            json.dump(features_filtered, f, indent=4)
    else:
        print(f"GRAIL posthoc experiment already exists in {ex_dir}.")

    # Test biomarkers ----------------------------------------------------------
    exname = 'test_biomarkers'
    ex_dir = os.path.join(output_dir, 'test_biomarkers')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['all_rsn_conns'] = False
        config_updates['n_pca_components'] = 10
        run(exname, observer, config_updates)
    else:
        print(f"Biomarkers experiment already exists in {ex_dir}.")

    # Fixed connection density ------------------------------------------------
    exname = 'test_and_finetune'
    ex_dir = os.path.join(output_dir, 'fixed_connection_density')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['num_epochs'] = 0 # no finetuning

        # Change edge transformation
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['dataset']['edge_tfm_type'] = 'DensityThresholdAdjacency'
        config_updates['dataset']['edge_tfm_params'] = {'density': 0.2,
                                                        'edge_info': 'functional_connectivity'}

        run(exname, observer, config_updates)
    else:
        print(f"Fixed connection density experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python graphtrip.py -c experiments/configs/graphtrip.json -o outputs/ -s 291 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/graphtrip/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
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
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
