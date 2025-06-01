"""
This scripts trains X-graphTRIP models for many train-test splits.
T-learners must be trained first with tlearners.py.

Dependencies:
- experiments/configs/x_graphtrip.json
- experiments/configs/x_graphtrip/tlearner_escitalopram/
- experiments/configs/x_graphtrip/tlearner_psilocybin/

Outputs:
- outputs/x_graphtrip/xlearner/job_{fold_shift}/

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
import json
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run
from utils.statsalg import pls_robustness_analysis, load_fold_data, compute_performance_weighted_means, pls_feature_filtering


def main(config_dir, output_dir, verbose, debug, seed, jobid=0, max_jobid=42):
    # Add project root to paths
    config_dir = add_project_root(config_dir)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    config_file = 'x_graphtrip.json'
    if not os.path.exists(os.path.join(config_dir, config_file)):
        raise FileNotFoundError(f"{config_file} not found in {config_dir}")
    
    # Load the config
    config = load_configs_from_json(os.path.join(config_dir, config_file))
    config = fetch_job_config(config, 0)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    # Train X-graphTRIP ---------------------------------------------------------
    exname = 'train_xlearner'
    weights_dir = os.path.join(output_dir, 'weights', f'job_{jobid}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['dataset']['fold_shift'] = jobid
        run(exname, observer, config_updates)
    else:
        print(f"X-graphTRIP experiment already exists in {weights_dir}.")

    # Train drug classifier -----------------------------------------------------
    exname = 'train_rep_classifier'
    ex_dir = os.path.join(output_dir, 'drug_classifier', f'job_{jobid}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir

        # Remove incompatible configs
        del config_updates['alpha']
        del config_updates['t0_pred_file']
        del config_updates['t1_pred_file']

        # Change MLP head and prediction target
        config_updates['mlp_model']['model_type'] = 'LogisticRegressionMLP'
        config_updates['dataset']['target'] = 'Condition_bin01'

        run(exname, observer, config_updates)
    else:
        print(f"Drug classifier experiment already exists in {weights_dir}.")

    # Attention weights ---------------------------------------------------------
    exname = 'attention_weights'
    ex_dir = os.path.join(output_dir, 'attention_weights', f'job_{jobid}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Attention weights experiment already exists in {ex_dir}.")

    if jobid == max_jobid:
        # Run PLS analysis on attention weights
        ex_dir = os.path.join(output_dir, 'attention_weights', 'pls_analysis')
        os.makedirs(ex_dir, exist_ok=True)

        # Define criteria for fold models to include in PLS analysis
        inclusion_criteria = {'metric': 'rho', 
                              'criterion': 'greater_than', 
                              'threshold': 0}
        
        # Save inclusion criteria to inclusion_criteria.json
        config_path = os.path.join(ex_dir, 'inclusion_criteria.json')
        with open(config_path, 'w') as f:
            json.dump(inclusion_criteria, f, indent=4)
        
        # Define job directories and filenames
        base_dir = os.path.join(output_dir, 'attention_weights')
        job_dirs = [os.path.join(base_dir, f'job_{j}') for j in range(max_jobid)]
        num_folds = config['dataset']['num_folds']
        filenames = [f'k{k}_attention_weights_original.csv' for k in range(num_folds)]

        # Aggregate the results for each subject from all job directories
        all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)

        # Run PLS analysis
        pls_patterns, pls_performance_corrs, pls_weight_stats = \
            pls_robustness_analysis(all_subject_dfs, fold_performances, 
                                    fold_performances[inclusion_criteria['metric']].values)
        
        # Save the results
        pls_patterns.to_csv(os.path.join(ex_dir, 'pls_patterns.csv'), index=False)
        pls_performance_corrs.to_csv(os.path.join(ex_dir, 'pls_performance_corrs.csv'), index=False)
        pls_weight_stats.to_csv(os.path.join(ex_dir, 'pls_weight_stats.csv'), index=False)
    else:
        print(f"PLS analysis on attention weights already exists in {ex_dir}.")

    # GRAIL --------------------------------------------------------------------
    exname = 'grail'
    ex_dir = os.path.join(output_dir, 'grail', f'job_{jobid}')
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

    if jobid == max_jobid:
        # Run PLS analysis on GRAIL results
        ex_dir = os.path.join(output_dir, 'grail', 'pls_analysis')
        os.makedirs(ex_dir, exist_ok=True)

        # Define criteria for fold models to include in PLS analysis
        inclusion_criteria = {'metric': 'rho', 
                              'criterion': 'greater_than', 
                              'threshold': 0}
        
        # Save inclusion criteria to inclusion_criteria.json
        config_path = os.path.join(ex_dir, 'inclusion_criteria.json')
        with open(config_path, 'w') as f:
            json.dump(inclusion_criteria, f, indent=4)
        
        # Define job directories and filenames
        base_dir = os.path.join(output_dir, 'grail')
        job_dirs = [os.path.join(base_dir, f'job_{j}') for j in range(max_jobid)]
        num_folds = config['dataset']['num_folds']
        filenames = [f'k{k}_mean_alignments.csv' for k in range(num_folds)]

        # Aggregate the results for each subject from all job directories
        all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)
        performance = fold_performances[inclusion_criteria['metric']].values

        # Run PLS analysis
        pls_patterns, pls_performance_corrs, pls_weight_stats = \
            pls_robustness_analysis(all_subject_dfs, performance)
        
        # Save the results
        pls_patterns.to_csv(os.path.join(ex_dir, 'pls_patterns.csv'), index=False)
        pls_performance_corrs.to_csv(os.path.join(ex_dir, 'pls_performance_corrs.csv'), index=False)
        pls_weight_stats.to_csv(os.path.join(ex_dir, 'pls_weight_stats.csv'), index=False)

        # Compute performance-weighted means
        weighted_means, weighted_means_stats = \
            compute_performance_weighted_means(all_subject_dfs, performance)

        # Filter candidate biomarkers:
        # 1. Significant PLS weight (fdr < 0.05) with large effect size (abs(cohen's d) > 0.8)
        # 2. Significant performance-weighted mean (fdr < 0.05)
        # 3. Mean PLS weight and weighted-mean should have the same sign
        filter_criteria = {
            'pls_weight_stats': {'fdr_p_value': 0.05, 'cohen_d': 0.8},
            'weighted_means_stats': {'fdr_p_value': 0.05},
            'signed_features': True
        }
        
        filtered_features = pls_feature_filtering(pls_weight_stats, 
                                                  weighted_means_stats, 
                                                  filter_criteria)

        # Save filter criteria to filter_criteria.json
        config_path = os.path.join(ex_dir, 'filtering_results.json')
        filter_criteria['filtered_features'] = filtered_features
        with open(config_path, 'w') as f:
            json.dump(filter_criteria, f, indent=4)
    else:
        print(f"PLS analysis on GRAIL results already exists in {ex_dir}.")
    

if __name__ == "__main__":
    """
    How to run:
    python x_graphtrip_training.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg -j 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_dir', type=str, default='experiments/configs/', help='Path to the config directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--jobid', type=int, default=0, help='Job ID')
    parser.add_argument('-mxj', '--max_jobid', type=int, default=41, 
                        help='Maximum possible job ID for this script. Typically the dataset size.')
    args = parser.parse_args()

    # Make sure debugging outputs don't overwrite any existing outputs
    if args.debug:
        if args.output_dir == 'outputs/x_graphtrip/':
            raise ValueError("output_dir must be specified when using debug mode.")

    # Run the main function
    main(args.config_dir, args.output_dir, args.verbose, args.debug, args.seed, args.jobid, args.max_jobid)
