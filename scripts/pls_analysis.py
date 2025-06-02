"""
This scripts performs PLS robustness analysis for the interpretability
results of the X-graphTRIP model. It requires the outputs of the 
x_graphtrip.py script.

Dependencies:
- outputs/x_graphtrip/grail/job_{fold_shift}/
- outputs/x_graphtrip/attention_weights/job_{fold_shift}/

Outputs:
- outputs/x_graphtrip/grail/pls_analysis/
- outputs/x_graphtrip/attention_weights/pls_analysis/

Author: Hanna M. Tolle
Date: 2025-06-02
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import json
import glob
import argparse
from utils.files import add_project_root
from utils.statsalg import pls_robustness_analysis, load_fold_data, compute_performance_weighted_means, pls_feature_filtering


def main(grail_dir, attention_dir, output_dir_name):
    # Add project root to paths
    grail_dir = add_project_root(grail_dir) if grail_dir is not None else None
    attention_dir = add_project_root(attention_dir) if attention_dir is not None else None

    # Make sure the directories exist
    if grail_dir is not None and not os.path.exists(grail_dir):
        raise FileNotFoundError(f"GRAIL results directory not found: {grail_dir}")
    if attention_dir is not None and not os.path.exists(attention_dir):
        raise FileNotFoundError(f"Attention weights results directory not found: {attention_dir}")
    
    # Attention weights ---------------------------------------------------------
    if attention_dir is not None:
        ex_dir = os.path.join(attention_dir, output_dir_name)

        if not os.path.exists(ex_dir):
            os.makedirs(ex_dir, exist_ok=True)

            # Define criteria for fold models to include in PLS analysis
            inclusion_criteria = {'metric': 'rho', 
                                  'criterion': 'greater_than', 
                                  'threshold': 0}
            
            # Save inclusion criteria to inclusion_criteria.json
            config_path = os.path.join(ex_dir, 'inclusion_criteria.json')
            with open(config_path, 'w') as f:
                json.dump(inclusion_criteria, f, indent=4)
            
            # Get all job directories and sort by job index
            job_dirs = sorted([d for d in glob.glob(os.path.join(attention_dir, 'job_*'))], 
                            key=lambda x: int(x.split('_')[-1]))
            
            # Get the number of folds from the first job directory
            fold_files = [f for f in os.listdir(job_dirs[0]) if f.startswith('k') and f.endswith('_attention_weights_original.csv')]
            num_folds = len(fold_files)
            filenames = [f'k{k}_attention_weights_original.csv' for k in range(num_folds)]

            # Aggregate the results for each subject from all job directories
            all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)
            performance = fold_performances[inclusion_criteria['metric']].values

            # Run PLS analysis
            pls_patterns, pls_performance_corrs, pls_weight_stats = \
                pls_robustness_analysis(all_subject_dfs, performance)
            
            # Save the results
            pls_patterns.to_csv(os.path.join(ex_dir, 'pls_patterns.csv'), index=False)
            pls_performance_corrs.to_csv(os.path.join(ex_dir, 'pls_performance_corrs.csv'), index=False)
            pls_weight_stats.to_csv(os.path.join(ex_dir, 'pls_weight_stats.csv'), index=True)

            print(f"PLS analysis on attention weights completed. Output saved in {ex_dir}.")
        else:
            print(f"PLS analysis on attention weights already exists in {ex_dir}.")

    # GRAIL --------------------------------------------------------------------
    if grail_dir is not None:
        ex_dir = os.path.join(grail_dir, output_dir_name)
        if not os.path.exists(ex_dir):
            os.makedirs(ex_dir, exist_ok=True)

            # Define criteria for fold models to include in PLS analysis
            inclusion_criteria = {'metric': 'rho', 
                                'criterion': 'greater_than', 
                                'threshold': 0}
            
            # Save inclusion criteria to inclusion_criteria.json
            config_path = os.path.join(ex_dir, 'inclusion_criteria.json')
            with open(config_path, 'w') as f:
                json.dump(inclusion_criteria, f, indent=4)
            
            # Get all job directories and sort by job index
            job_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'job_*'))], 
                            key=lambda x: int(x.split('_')[-1]))
            
            # Get the number of folds from the first job directory
            fold_files = [f for f in os.listdir(job_dirs[0]) if f.startswith('k') and f.endswith('_mean_alignments.csv')]
            num_folds = len(fold_files)
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
            pls_weight_stats.to_csv(os.path.join(ex_dir, 'pls_weight_stats.csv'), index=True)

            # Compute performance-weighted means
            weighted_means, weighted_means_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance)
            
            # Save the performance-weighted means
            weighted_means.to_csv(os.path.join(ex_dir, 'weighted_means.csv'), index=False)
            weighted_means_stats.to_csv(os.path.join(ex_dir, 'weighted_means_stats.csv'), index=True)

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

            print(f"PLS analysis on GRAIL results completed. Output saved in {ex_dir}.")
        else:
            print(f"PLS analysis on GRAIL results already exists in {ex_dir}.")
    

if __name__ == "__main__":
    """
    How to run:
    python pls_analysis.py --grail_dir outputs/x_graphtrip/grail/ --attention_dir outputs/x_graphtrip/attention_weights/ --output_dir_name pls_analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--grail_dir', type=str, default=None, help='Path to the GRAIL results directory')
    parser.add_argument('--attention_dir', type=str, default=None, help='Path to the attention weights results directory')
    parser.add_argument('-o', '--output_dir_name', type=str, default='pls_analysis', help='Name of the output directory, created inside the grail or attention weights directory')
    args = parser.parse_args()

    # At least one of the directories must be provided
    if args.grail_dir is None and args.attention_dir is None:
        raise ValueError("At least one of the directories must be provided.")

    # Run the main function
    main(args.grail_dir, args.attention_dir, args.output_dir_name)
