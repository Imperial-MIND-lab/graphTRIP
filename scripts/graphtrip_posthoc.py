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
from utils.files import add_project_root
from utils.configs import load_configs_from_json
from utils.statsalg import grail_posthoc_analysis, load_fold_data, compute_performance_weighted_means, elasticnet_cv_predict
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
    ex_dir = os.path.join(output_dir, 'grail_posthoc')
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir, exist_ok=True)
        grail_dir = os.path.join(output_dir, 'grail')
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


if __name__ == "__main__":
    """
    How to run:
    python graphtrip_posthoc.py -w experiments/outputs/graphtrip/weights/ -o outputs/ -s 0 -v
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_base_dir', type=str, 
                        default='outputs/graphtrip/weights/', 
                        help='Path to the base directory with graphTRIP weights')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='outputs/graphtrip/', 
                        help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Run the main function
    main(args.weights_base_dir, args.output_dir, args.verbose, args.seed)
