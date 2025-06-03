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

import sys
sys.path.append("../")

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import sort_features
from utils.plotting import plot_diverging_bars, plot_diverging_raincloud, plot_histogram
from utils.plotting import custom_diverging_cmap, PSILO, ESCIT, COOLWARM
from utils.statsalg import pls_robustness_analysis, load_fold_data, compute_performance_weighted_means, pls_feature_filtering
from preprocessing.metrics import get_rsn_mapping


def main(grail_dir=None, attention_dir=None, output_dir_name='pls_analysis', 
         criteria=None, overwrite=False):
    # Add project root to paths
    grail_dir = add_project_root(grail_dir) if grail_dir is not None else None
    attention_dir = add_project_root(attention_dir) if attention_dir is not None else None

    # Make sure the directories exist
    if grail_dir is not None and not os.path.exists(grail_dir):
        raise FileNotFoundError(f"GRAIL results directory not found: {grail_dir}")
    if attention_dir is not None and not os.path.exists(attention_dir):
        raise FileNotFoundError(f"Attention weights results directory not found: {attention_dir}")
    
    # Define default inclusion and filtering criteria
    if criteria is None:
        inclusion_criteria = {'metric': 'rho', 'criterion': 'greater_than', 'threshold': 0}
        filtering_criteria = {'pls_weight_stats': {'fdr_p_value': 0.05, 'cohen_d': 0.8},
                              'weighted_means_stats': {'fdr_p_value': 0.05},
                              'signed_features': True}
    else:
        inclusion_criteria = criteria.get('inclusion', {'metric': 'rho', 'criterion': 'greater_than', 'threshold': 0})
        filtering_criteria = criteria.get('filtering', {'pls_weight_stats': {'fdr_p_value': 0.05, 'cohen_d': 0.8},
                                                        'weighted_means_stats': {'fdr_p_value': 0.05},
                                                        'signed_features': True})
    
    # Attention weights ---------------------------------------------------------
    if attention_dir is not None:
        ex_dir = os.path.join(attention_dir, output_dir_name)

        if not os.path.exists(ex_dir) or overwrite:
            save_figs = True
            os.makedirs(ex_dir, exist_ok=True)
            
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

            # Infer the brain atlas from config.json inside the job directory
            config_path = os.path.join(job_dirs[0], 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            atlas = config['dataset']['atlas']

            # Compute mean attention within resting-state networks
            rsn_mapping, rsn_names = get_rsn_mapping(atlas)
            mean_rsn_pls = {rsn: [] for rsn in rsn_names}
            for rsn_idx, rsn_name in enumerate(rsn_names):
                # Get the columns for this RSN
                rsn_columns = pls_patterns.columns[rsn_mapping == rsn_idx]

                # Average across RSN columns to get the mean attention for this RSN for each subject
                rsn_pls = pls_patterns[rsn_columns].mean(axis=1)
                mean_rsn_pls[rsn_name] = list(rsn_pls)

            # Save mean RSN attention
            mean_rsn_pls_df = pd.DataFrame(mean_rsn_pls)
            mean_rsn_pls_df.to_csv(os.path.join(ex_dir, 'mean_rsn_attention.csv'), index=False)

            print(f"PLS analysis on attention weights completed. Output saved in {ex_dir}.")
        else:
            save_figs = False
            print(f"PLS analysis on attention weights already exists in {ex_dir}.")

            # Load the results for plotting
            mean_rsn_pls = pd.read_csv(os.path.join(ex_dir, 'mean_rsn_attention.csv'))
            mean_rsn_pls = mean_rsn_pls.to_dict(orient='list')
            pls_performance_corrs = pd.read_csv(os.path.join(ex_dir, 'pls_performance_corrs.csv'))

        # Plot results ------------------------------------------------------------
        mean_rsn_pls = {k: v for k, v in sorted(mean_rsn_pls.items(), key=lambda item: np.mean(item[1]), reverse=True)}
        vmax = np.percentile(abs(np.array(list(mean_rsn_pls.values())).flatten()), 75)
        save_path = os.path.join(ex_dir, 'mean_rsn_attention.png') if save_figs else None
        plot_diverging_raincloud(mean_rsn_pls, 
                                cmap=COOLWARM, 
                                vmax=vmax, 
                                alpha=0.7, 
                                box_alpha=0.7, 
                                scatter_alpha=0.5,
                                save_path=save_path, 
                                figsize=(5, 5), 
                                add_asterisk=True, 
                                add_colorbar=False);

        # Plot histogram of correlation (r) values of PLS components with fold-model performance
        save_path = os.path.join(ex_dir, 'pls_correlations_histogram.png') if save_figs else None
        corr_values = np.abs(pls_performance_corrs['r'].values)

        # Calculate statistics and add to title
        r_mean = np.mean(corr_values)
        r_sem = np.std(corr_values) / np.sqrt(len(corr_values))
        percentage_of_significant_rs = (np.sum(pls_performance_corrs['p'] < 0.05) / len(pls_performance_corrs['p']))*100
        title = f'r = {r_mean:.4f} ± {r_sem:.4f}; % significant = {percentage_of_significant_rs:.2f}%'
        plot_histogram(distributions = {'r': corr_values}, 
                       palette={'r': 'darkblue'}, 
                       alpha=0.6, 
                       save_path=save_path, 
                       figsize=(5, 4),
                       title=title)

    # GRAIL --------------------------------------------------------------------
    if grail_dir is not None:
        ex_dir = os.path.join(grail_dir, output_dir_name)
        if not os.path.exists(ex_dir) or overwrite:
            save_figs = True
            os.makedirs(ex_dir, exist_ok=True)
            
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
            
            # Filter candidate biomarkers
            filtered_features = pls_feature_filtering(pls_weight_stats, 
                                                      weighted_means_stats, 
                                                      filtering_criteria)

            # Sort features based category and weighted-mean sign
            features_presorted = sort_features(filtered_features)
            features_sorted = [f for f in features_presorted if weighted_means[f].mean() < 0] + \
                              [f for f in features_presorted if weighted_means[f].mean() > 0] 
            
            # Save filter criteria and sorted features to filter_criteria.json
            config_path = os.path.join(ex_dir, 'filtering_results.json')
            filtering_criteria['filtered_features'] = features_sorted
            with open(config_path, 'w') as f:
                json.dump(filtering_criteria, f, indent=4)

            print(f"PLS analysis on GRAIL results completed. Output saved in {ex_dir}.")
        else:
            save_figs = False
            print(f"PLS analysis on GRAIL results already exists in {ex_dir}.")

            # Load the results for plotting
            weighted_means = pd.read_csv(os.path.join(ex_dir, 'weighted_means.csv'))
            filtering_results = json.load(open(os.path.join(ex_dir, 'filtering_results.json')))
            features_sorted = filtering_results['filtered_features']
            pls_performance_corrs = pd.read_csv(os.path.join(ex_dir, 'pls_performance_corrs.csv'))

        # Plot results ------------------------------------------------------------
        psilo_escit_cmap = custom_diverging_cmap(PSILO, ESCIT, n_colors=256)
        vmax = np.percentile(abs(weighted_means[features_sorted].values), 75)
        barwidth = 0.65
        save_path = os.path.join(ex_dir, 'pls_grail_features.png')
        fig_size = (min(13, len(features_sorted)*barwidth), 5)
        fig, ax = plot_diverging_bars(weighted_means[features_sorted], 
                            yline=0, 
                            cmap=psilo_escit_cmap, 
                            vmax=vmax, 
                            alpha=0.6,
                            add_scatter=True, 
                            scatter_alpha=0.5, 
                            scatter_size=20,
                            figsize=fig_size,
                            save_path=None, # Don't save yet
                            add_colorbar=False)
        
        # Add padding below plot for x-tick labels
        if save_figs:
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(save_path, bbox_inches='tight')

        # Plot histogram of correlation (r) values of PLS components with fold-model performance
        save_path = os.path.join(ex_dir, 'pls_correlations_histogram.png') if save_figs else None
        corr_values = np.abs(pls_performance_corrs['r'].values)

        # Calculate statistics and add to title
        r_mean = np.mean(corr_values)
        r_sem = np.std(corr_values) / np.sqrt(len(corr_values))
        percentage_of_significant_rs = (np.sum(pls_performance_corrs['p'] < 0.05) / len(pls_performance_corrs['p']))*100
        title = f'r = {r_mean:.4f} ± {r_sem:.4f}; % significant = {percentage_of_significant_rs:.2f}%'
        plot_histogram(distributions = {'r': corr_values}, 
                       palette={'r': 'darkblue'}, 
                       alpha=0.6, 
                       save_path=save_path, 
                       figsize=(5, 4),
                       title=title)


if __name__ == "__main__":
    """
    How to run:
    python pls_analysis.py --grail_dir outputs/x_graphtrip/grail/ --attention_dir outputs/x_graphtrip/attention_weights/ --output_dir_name pls_analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--grail_dir', type=str, default=None, help='Path to the GRAIL results directory')
    parser.add_argument('--attention_dir', type=str, default=None, help='Path to the attention weights results directory')
    parser.add_argument('-o', '--output_dir_name', type=str, default='pls_analysis', \
                        help='Name of the output directory, created inside the grail or attention weights directory')
    args = parser.parse_args()

    # At least one of the directories must be provided
    if args.grail_dir is None and args.attention_dir is None:
        raise ValueError("At least one of the directories must be provided.")

    # Run the main function
    main(args.grail_dir, args.attention_dir, args.output_dir_name)
