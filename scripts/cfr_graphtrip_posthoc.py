"""
This scripts performs all post-hoc analyses for graphTRIP.

Dependencies:
- experiments/outputs/weights/seed_*/
- experiments/outputs/graphtrip/grail/seed_*/

Outputs:
- outputs/graphtrip/grail_posthoc/
- outputs/graphtrip/test_biomarkers/

Author: Hanna M. Tolle
Date: 2025-11-18
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import copy
import argparse
import glob
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
from utils.files import add_project_root
from utils.configs import load_configs_from_json
from utils.statsalg import grail_posthoc_analysis, load_fold_data, compute_performance_weighted_means, elasticnet_cv_predict
from utils.statsalg import regional_permutation_cluster_test
from preprocessing.metrics import compute_dist_adjacency
from experiments.run_experiment import run


def main(weights_base_dir, output_dir, verbose, seed):

    # Add project root to paths
    weights_base_dir = add_project_root(weights_base_dir)
    weights_dirs = sorted([d for d in glob.glob(os.path.join(weights_base_dir, 'seed_*'))
                           if os.path.isdir(d)])
    config = load_configs_from_json(os.path.join(weights_dirs[0], 'config.json'))
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config_updates = {}
    config_updates['verbose'] = verbose
    config_updates['seed'] = seed

    # Output directory
    output_dir = add_project_root(output_dir)

    # GRAIL posthoc ----------------------------------------------------------
    drug_conditions = ["escitalopram", "psilocybin"]
    for drug_condition in drug_conditions:
        ex_dir = os.path.join(output_dir, 'grail_posthoc', drug_condition)
        if not os.path.exists(ex_dir):
            os.makedirs(ex_dir, exist_ok=True)
            grail_dir = os.path.join(output_dir, 'grail', drug_condition)
            assert os.path.exists(grail_dir), f"GRAIL directory {grail_dir} not found"

            # CV-fold model inclusion criteria
            metric = 'r' # select models based on true-vs-pred test correlation
            inclusion_criteria = {'metric': metric, 
                                'criterion': 'greater_than', 
                                'threshold': 0.00} # only consider positive corrs

            # Settings for clustering GRAIL results (into subtypes)
            filter_percentile = 75 # alignment magnitude pctl to consider as relevant
            num_seeds = 10         # number of seeds for clustering

            # Save all analysis settings
            grail_posthoc_config = {
                    'inclusion_criteria': inclusion_criteria,
                    'filter_percentile': filter_percentile,
                    'num_seeds': num_seeds,
                    'seed': seed}
            with open(os.path.join(ex_dir, 'config.json'), 'w') as f:
                json.dump(grail_posthoc_config, f, indent=4)

            # Generate filenames for k-fold mean alignments
            num_folds = config['dataset']['num_folds']
            filenames = [f'k{k}_mean_alignments.csv' for k in range(num_folds)]
            seed_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'seed_*'))
                                if os.path.isdir(d)])

            # Load subject GRAIL dfs and fold model performance
            all_subject_dfs, fold_performances = load_fold_data(seed_dirs, filenames, inclusion_criteria)
            performance = fold_performances[metric].values

            # Compute performance-weighted mean alignments and stats
            weighted_mean_alignments, weighted_mean_alignments_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance)    

            # Run clustering analysis and filter biomarkers
            cluster_labels, features_filtered = grail_posthoc_analysis(weighted_mean_alignments,
                                                                    num_seeds, seed,
                                                                    filter_percentile)
            # Save results
            np.savetxt(os.path.join(ex_dir, 'cluster_labels.csv'), cluster_labels, delimiter=',')
            with open(os.path.join(ex_dir, 'features_filtered.json'), 'w') as f:
                json.dump(features_filtered, f, indent=4)
            
            weighted_mean_alignments_stats.to_csv(os.path.join(ex_dir, 'weighted_mean_alignments_stats.csv'))
            weighted_mean_alignments.to_csv(os.path.join(ex_dir, 'weighted_mean_alignments.csv'))

            print(f"GRAIL posthoc analysis completed. Output saved in {ex_dir}.")
        else:
            print(f"GRAIL posthoc experiment already exists in {ex_dir}.")

    # Test biomarkers ----------------------------------------------------------
    exname = 'test_biomarkers'
    ex_dir = os.path.join(output_dir, 'test_biomarkers')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dirs[0]
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['all_rsn_conns'] = False
        config_updates['n_pca_components'] = 10
        run(exname, observer, config_updates)

        # Benchmark graphTRIP against feature-based linear regression
        feature_values = pd.read_csv(os.path.join(ex_dir, 'feature_values.csv'))
        y = feature_values['y'].values
        feature_names = feature_values.columns.drop(['y', 'sub'])
        X = feature_values[feature_names].values
        df = elasticnet_cv_predict(X, y, subject_ids=feature_values['sub'].values, n_splits=7, random_state=0)
        df.to_csv(os.path.join(ex_dir, 'elasticnet_cv_predict.csv'), index=False)
    else:
        print(f"Biomarkers experiment already exists in {ex_dir}.")

    # Latent node importance posthoc analysis -----------------------------------
    drug_conditions = ["escitalopram", "psilocybin"]
    ex_dir = os.path.join(output_dir, 'latent_node_importance_posthoc')
    os.makedirs(ex_dir, exist_ok=True)
    for drug_condition in drug_conditions:
        ex_base_dir = os.path.join(output_dir, 'latent_node_importance', drug_condition)
        assert os.path.exists(ex_base_dir), f"Latent node importance directory {ex_base_dir} not found"

        # Analysis config
        metric = 'r'     # select models based on true-vs-pred test correlation
        threshold = 0.00 # only consider positive corrs
        criterion = 'greater_than'
        
        # Get all seed directories
        seed_dirs = sorted([d for d in glob.glob(os.path.join(ex_base_dir, 'seed_*')) if os.path.isdir(d)])
        
        # Load node importance scores and fold performance from all seeds
        node_importance_list = []
        fold_performance_list = []
        for seed_idx, seed_dir in enumerate(seed_dirs):
            # Load node importance scores (fold is index, regions are columns)
            node_imp_df = pd.read_csv(os.path.join(seed_dir, 'node_importance_scores.csv'))
            node_imp_df['seed_fold_id'] = f"seed_{seed_idx}_fold_" + node_imp_df['fold'].astype(str)
            node_importance_list.append(node_imp_df)
            
            # Load fold performance (fold is a column)
            fold_perf_df = pd.read_csv(os.path.join(seed_dir, 'fold_performance.csv'))
            fold_perf_df['seed_fold_id'] = f"seed_{seed_idx}_fold_" + fold_perf_df['fold'].astype(str)
            fold_performance_list.append(fold_perf_df)
        
        # Concatenate all node importance scores and fold performance
        node_importance_scores = pd.concat(node_importance_list, ignore_index=True).drop(columns=['fold'])
        fold_performance = pd.concat(fold_performance_list, ignore_index=True).drop(columns=['fold'])
        
        # Merge node importance scores with fold performance on 'seed_fold_id' to ensure correct matching
        merged_df = node_importance_scores.merge(fold_performance[['seed_fold_id', metric]], on='seed_fold_id', how='inner')
        
        # Filter rows based on inclusion criteria
        if criterion == 'greater_than':
            filtered_mask = merged_df[metric] > threshold
        elif criterion == 'less_than':
            filtered_mask = merged_df[metric] < threshold
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        filtered_df = merged_df[filtered_mask].copy()
        
        if len(filtered_df) == 0:
            print(f"Warning: No fold models passed inclusion criteria for {drug_condition}. Skipping.")
            continue
        
        # Get region names (all columns except metadata columns)
        region_columns = [col for col in filtered_df.columns if col not in [metric, 'seed_fold_id']]

        # Extract performance values and node importance scores
        performance = filtered_df[metric].values          # shape: (num_filtered_folds,)
        node_scores = filtered_df[region_columns].values  # shape: (num_filtered_folds, num_regions)
        
        # Compute performance-weighted mean (similar to compute_performance_weighted_means)
        # Set negative performance values to 0
        performance_cutoff = 0.0
        performance_weights = performance.copy()
        performance_weights[performance_weights < performance_cutoff] = 0
        weights = performance_weights / (np.sum(performance_weights) + 1e-6)  # shape: (num_filtered_folds,)
        
        # Compute weighted mean: sum(weights[:, None] * node_scores, axis=0)
        weighted_mean_scores = np.sum(weights[:, np.newaxis] * node_scores, axis=0)  # shape: (num_regions,)
        
        # Create single-row DataFrame with region names as columns
        weighted_mean_node_importance = pd.DataFrame(
            weighted_mean_scores.reshape(1, -1),
            columns=region_columns)
        
        # Save results
        output_file = os.path.join(ex_dir, f'{drug_condition}_weighted_mean_node_importance.csv')
        weighted_mean_node_importance.to_csv(output_file, index=False)
        
        print(f"Latent node importance posthoc analysis for {drug_condition} completed.")
        print(f"  - Included {len(filtered_df)} fold models out of {len(merged_df)} total")
        print(f"  - Output saved to {output_file}")


if __name__ == "__main__":
    """
    How to run:
    python graphtrip_posthoc.py -w experiments/outputs/graphtrip/weights/ -o outputs/ -s 0 -v
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_base_dir', type=str, 
                        default='outputs/cfr_graphtrip/weights/', 
                        help='Path to the base directory with graphTRIP weights')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='outputs/graphtrip/', 
                        help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Run the main function
    main(args.weights_base_dir, args.output_dir, args.verbose, args.seed)
