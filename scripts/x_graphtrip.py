"""
This scripts trains X-graphTRIP models for many train-test splits.
T-learners must be trained first with tlearners.py.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/tlearner.json

Outputs:
- outputs/x_graphtrip/vgae_weights/seed_{seed}/test_fold_indices.csv
- outputs/x_graphtrip/estimate_propensity/seed_{seed}/
- outputs/x_graphtrip/tlearner/seed_{seed}/
- outputs/x_graphtrip/cate_model/seed_{seed}/

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


def main(config_file, tlearner_config_file, output_dir, verbose, debug, seed, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    tlearner_config_file = add_project_root(tlearner_config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    if not os.path.exists(tlearner_config_file):
        raise FileNotFoundError(f"{tlearner_config_file} not found")
    
    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
    tlearner_config = load_configs_from_json(tlearner_config_file)
    tlearner_config = fetch_job_config(tlearner_config, config_id)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    if debug:
        config['num_epochs'] = 2
        tlearner_config['num_epochs'] = 2
    tlearner_config['seed'] = seed
    tlearner_config['verbose'] = verbose

    # Train VGAEs (unsupervised training) ---------------------------------------
    exname = 'train_vgae'
    vgae_weights_dir = os.path.join(output_dir, 'vgae_weights', f'seed_{seed}')

    # If test_fold_indices.csv exists, then it has already been evaluated
    if not os.path.exists(vgae_weights_dir):
        print(f"Training VGAEs (unsupervised training) in {vgae_weights_dir}.")

        # Get dataset and VGAE configs from original graphTRIP config
        config_updates = {}
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['vgae_model'] = copy.deepcopy(config['vgae_model'])

        # Remove labels from dataset config
        config_updates['dataset']['target'] = None
        config_updates['dataset']['graph_attrs'] = ['Condition'] # for balancing k-fold splits
        config_updates['dataset']['context_attrs'] = []          # pooling doesn't use context
        config_updates['this_k'] = None                          # train all folds sequentially

        # Remove pooling layer from VGAE config
        config_updates['vgae_model']['pooling_cfg'] = {
            'model_type': 'DummyPooling'}

        # Training configurations
        config_updates['lr'] = config['lr']
        config_updates['num_epochs'] = config['num_epochs']
        config_updates['balance_attrs'] = ['Condition']

        # Directories etc.
        config_updates['output_dir'] = vgae_weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['save_weights'] = True
        run(exname, observer, config_updates)
    else:
        print(f"VGAE weights already exist in {vgae_weights_dir}.")

    # Verify sufficient overlap -----------------------------------------------
    exname = 'estimate_propensity'
    ex_dir = os.path.join(output_dir, 'estimate_propensity', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        print(f"Training propensity model in {ex_dir}.")
        
        config_updates = {}
        config_updates['logit_C'] = 1.0          # standard logit regularization
        config_updates['n_pca_components'] = 0   # no PCA dimensionality reduction
        config_updates['reinit_pooling'] = True  # replace "DummyPooling"

        # Remove Condition from graph_attrs and set target to Condition_bin01
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        graph_attrs = config_updates['dataset']['graph_attrs']
        new_graph_attrs = [attr for attr in graph_attrs if attr != 'Condition']
        config_updates['dataset']['graph_attrs'] = new_graph_attrs
        config_updates['dataset']['target'] = 'Condition_bin01'
        config_updates['dataset']['context_attrs'] = [] # pooling doesn't use context

        # Specify new pooling config
        config_updates['vgae_model'] = copy.deepcopy(config['vgae_model'])
        config_updates['vgae_model']['pooling_cfg'] = {
            'model_type': 'MeanStdPooling'}

        # Directories etc.
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = vgae_weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Propensity model experiment already exists in {ex_dir}.")

    # Train Medusa T-learner ----------------------------------------------------
    exname = 'train_cfrnet'
    ex_dir = os.path.join(output_dir, 'tlearner', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        print(f"Training Medusa T-learner in {ex_dir}.")
        config_updates = copy.deepcopy(tlearner_config)
        config_updates['test_fold_indices_file'] = os.path.join(vgae_weights_dir, 'test_fold_indices.csv')
        config_updates['output_dir'] = ex_dir
        config_updates['save_weights'] = True
        run(exname, observer, config_updates)
    else:
        print(f"Medusa T-learner experiment already exists in {ex_dir}.")
    tlearner_dir = ex_dir

    # Train S-CATE model -------------------------------------------------------
    exname = 'train_scate'
    ex_dir = os.path.join(output_dir, 'cate_model', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['dataset'] = copy.deepcopy(config['dataset'])

        # S-CATE model settings
        config_updates['prediction_head_type'] = 'Ridge'
        config_updates['standardize_data'] = True
        config_updates['n_permutations'] = 1000        

        # ITE label settings
        config_updates['dataset']['target'] = None
        config_updates['t0_pred_file'] = os.path.join(tlearner_dir, 'counterfactual_predictions.csv')
        config_updates['t1_pred_file'] = os.path.join(tlearner_dir, 'counterfactual_predictions.csv')
        config_updates['dataset']['graph_attrs_to_standardise'] = []

        # Condition settings
        config_updates['annotations_file'] = 'data/raw/psilodep2/annotations.csv'
        config_updates['subject_id_col'] = 'Patient'
        config_updates['condition_specs'] = {'cond0': 'E', 'cond1': 'P'}

        # VGAE pooling config
        config_updates['vgae_model'] = {}
        config_updates['vgae_model']['pooling_cfg'] = {
            'model_type': 'MeanStdPooling'}

        # Output and VGAE weights directories
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = vgae_weights_dir
        config_updates['save_weights'] = True
        config_updates['verbose'] = verbose
        config_updates['seed'] = seed
        run(exname, observer, config_updates)
    else:
        print(f"CATE model already exists in {ex_dir}.")


if __name__ == "__main__":
    """
    How to run:
    python x_graphtrip.py 
       -c experiments/configs/graphtrip.json 
       -t experiments/configs/tlearner.json 
       -o outputs/x_graphtrip/ 
       -s 0 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with X-graphTRIP model config')
    parser.add_argument('-t', '--tlearner_config_file', type=str, 
                        default='experiments/configs/tlearner.json', 
                        help='Path to the config file with Medusa T-learner config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
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
    main(args.config_file, args.tlearner_config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
