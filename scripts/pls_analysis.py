"""
This scripts performs PLS sensitivity analysis for the interpretability
results of the X-graphTRIP model. 

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
import seaborn as sns
from scipy.stats import chisquare

from utils.files import add_project_root
from utils.helpers import sort_features
from utils.plotting import plot_diverging_bars, plot_raincloud
from utils.plotting import custom_diverging_cmap, PSILO, ESCIT
from utils.statsalg import pls_robustness_analysis, load_fold_data, compute_performance_weighted_means, pls_feature_filtering
from preprocessing.metrics import get_rsn_mapping


def main(grail_dir:str=None,                     # Path to the GRAIL results directory
         attention_dir:str=None,                 # Path to the attention weights results directory
         output_dir_name:str='pls_analysis',     # Name of the output directory, created inside the grail or attention weights directory
         metric:str='rho',                       # Performance metric to use for the PLS analysis (rho or r)
         overwrite:bool=False):                  # Whether to overwrite the output directory if it already exists
    
    # Add project root to paths
    grail_dir = add_project_root(grail_dir) if grail_dir is not None else None
    attention_dir = add_project_root(attention_dir) if attention_dir is not None else None

    # Make sure the directories exist
    if grail_dir is not None and not os.path.exists(grail_dir):
        raise FileNotFoundError(f"GRAIL results directory not found: {grail_dir}")
    if attention_dir is not None and not os.path.exists(attention_dir):
        raise FileNotFoundError(f"Attention weights results directory not found: {attention_dir}")
    
    # Metric must be rho or r
    if metric not in ['rho', 'r']:
        raise ValueError(f"Metric must be rho or r, got {metric}")
    
    # Attention weights ---------------------------------------------------------
    if attention_dir is not None:
        ex_dir = os.path.join(attention_dir, output_dir_name)

        if not os.path.exists(ex_dir) or overwrite:
            save_figs = True
            os.makedirs(ex_dir, exist_ok=True)
            
            # Only include significant models
            inclusion_criteria = {'metric': 'p' if metric == 'r' else 'rho_p', 
                                  'criterion': 'less_than', 
                                  'threshold': 0.05}
            
            # Get all job directories and sort by job index
            job_dirs = sorted([d for d in glob.glob(os.path.join(attention_dir, 'job_*'))], 
                            key=lambda x: int(x.split('_')[-1]))
            
            # Get the number of folds from the first job directory
            fold_files = [f for f in os.listdir(job_dirs[0]) if f.startswith('k') and f.endswith('_attention_weights_original.csv')]
            num_folds = len(fold_files)
            filenames = [f'k{k}_attention_weights_original.csv' for k in range(num_folds)]

            # Aggregate the results for each subject from all job directories
            all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)
            assert len(fold_performances) > 0, "No significant models found."
            performance = fold_performances[metric].values

            # Compute performance-weighted means
            cutoff = 0.0 # make sure to only include significantly positive performance values
            weighted_means, weighted_means_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance, performance_cutoff=cutoff)
            
            # Save the results
            weighted_means.to_csv(os.path.join(ex_dir, 'weighted_means_significant.csv'), index=False)
            weighted_means_stats.to_csv(os.path.join(ex_dir, 'weighted_means_stats_significant.csv'), index=True)

            # Infer the brain atlas from config.json inside the job directory
            config_path = os.path.join(job_dirs[0], 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            atlas = config['dataset']['atlas']

            # Compute mean attention within resting-state networks
            rsn_mapping, rsn_names = get_rsn_mapping(atlas)
            mean_rsn_attention = {rsn: [] for rsn in rsn_names}
            for rsn_idx, rsn_name in enumerate(rsn_names):
                # Get the columns for this RSN
                rsn_columns = weighted_means.columns[rsn_mapping == rsn_idx]

                # Average across RSN columns to get the mean attention for this RSN for each subject
                rsn_mean = weighted_means[rsn_columns].mean(axis=1)
                mean_rsn_attention[rsn_name] = list(rsn_mean)

            # Save mean RSN attention
            mean_rsn_attention = pd.DataFrame(mean_rsn_attention)
            mean_rsn_attention.to_csv(os.path.join(ex_dir, 'weighted_mean_rsn_attention.csv'), index=False)

            print(f"PLS analysis on attention weights completed. Output saved in {ex_dir}.")
        else:
            save_figs = False
            print(f"PLS analysis on attention weights already exists in {ex_dir}.")

            # Load the results for plotting
            mean_rsn_attention = pd.read_csv(os.path.join(ex_dir, 'weighted_mean_rsn_attention.csv'))

        # Plot results ------------------------------------------------------------
        mean_rsn_attention = {k: v for k, v in sorted(mean_rsn_attention.items(), key=lambda item: np.mean(item[1]), reverse=True)}
        save_path = os.path.join(ex_dir, 'weighted_mean_rsn_attention.png') if save_figs else None
        offset = 3
        colors = sns.color_palette("YlGnBu_r", len(mean_rsn_attention)+offset)
        palette = {name: color for name, color in zip(mean_rsn_attention.keys(), colors)}
        plot_raincloud(mean_rsn_attention, 
                       palette=palette, 
                       save_path=save_path, 
                       alpha=0.7, 
                       box_alpha=0.5,
                       figsize=(5, 5),
                       sort_by_mean=False)

    # GRAIL --------------------------------------------------------------------
    if grail_dir is not None:
        ex_dir = os.path.join(grail_dir, output_dir_name)
        if not os.path.exists(ex_dir) or overwrite:
            save_figs = True
            os.makedirs(ex_dir, exist_ok=True)
                
            # Get all job directories and sort by job index
            job_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'job_*'))], 
                            key=lambda x: int(x.split('_')[-1]))
            
            # Get the number of folds from the first job directory
            fold_files = [f for f in os.listdir(job_dirs[0]) if f.startswith('k') and f.endswith('_mean_alignments.csv')]
            num_folds = len(fold_files)
            filenames = [f'k{k}_mean_alignments.csv' for k in range(num_folds)]

            # Only use significance threshold as filtering criterion
            filtering_criteria = {'pls_weight_stats': {'fdr_p_value': 0.05},     # significant influence on generalisation
                                  'weighted_means_stats': {'fdr_p_value': 0.05}, # necessary for generalisation
                                  'signed_features': True}                       # positive, not negative, influence on generalisation

            # Load subject GRAIL dfs and fold model performance
            inclusion_criteria = {'metric': metric, 
                                  'criterion': 'greater_than', 
                                  'threshold': 0.3}
            all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)
            performance = fold_performances[metric].values

            # Perform PLS analysis on all models meeting the inclusion criteria
            _, _, pls_weight_stats = pls_robustness_analysis(all_subject_dfs, performance)

            # Compute performance-weighted means
            weighted_means, weighted_means_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance)
                
            # Filter candidate biomarkers
            filtered_features = pls_feature_filtering(pls_weight_stats, 
                                                        weighted_means_stats, 
                                                        filtering_criteria)

            # Get the associations for each feature
            all_features = list(weighted_means.columns)
            eso_features = [f for f in filtered_features if weighted_means[f].mean() > 0]
            psilo_features = [f for f in filtered_features if weighted_means[f].mean() < 0]

            # Sort features and save as json
            all_features = sort_features(all_features)
            eso_features = sort_features(eso_features)
            psilo_features = sort_features(psilo_features)
            robust_features = {'all': all_features,
                               'escitalopram': eso_features, 
                               'psilocybin': psilo_features}
            with open(os.path.join(ex_dir, 'robust_features.json'), 'w') as f:
                json.dump(robust_features, f, indent=4)

            # Save pls weight stats and weighted means stats
            pls_weight_stats.to_csv(os.path.join(ex_dir, 'pls_weight_stats.csv'), index=True)
            weighted_means_stats.to_csv(os.path.join(ex_dir, 'weighted_means_stats.csv'), index=True)

            # --------------------------------------------------------------------------------------------------
            # Also compute and save the weighted means for significant models ----------------------------------
            # --------------------------------------------------------------------------------------------------
            inclusion_criteria = {'metric': 'p' if metric == 'r' else 'rho_p', 
                                  'criterion': 'less_than', 
                                  'threshold': 0.05}
            
            # Get all job directories and sort by job index
            job_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'job_*'))], 
                            key=lambda x: int(x.split('_')[-1]))
            
            # Get the number of folds from the first job directory
            fold_files = [f for f in os.listdir(job_dirs[0]) if f.startswith('k') and f.endswith('_mean_alignments.csv')]
            num_folds = len(fold_files)
            filenames = [f'k{k}_mean_alignments.csv' for k in range(num_folds)]

            # Aggregate the results for each subject from all job directories
            all_subject_dfs, fold_performances = load_fold_data(job_dirs, filenames, inclusion_criteria)
            assert len(fold_performances) > 0, "No significant models found."
            performance = fold_performances[metric].values

            # Compute performance-weighted means
            cutoff = 0.0 # make sure to only include significantly positive performance values
            weighted_means_significant, weighted_means_stats_significant = \
                compute_performance_weighted_means(all_subject_dfs, performance, performance_cutoff=cutoff)
            
            # Save the results
            weighted_means_significant.to_csv(os.path.join(ex_dir, 'weighted_means_significant.csv'), index=False)
            weighted_means_stats_significant.to_csv(os.path.join(ex_dir, 'weighted_means_stats_significant.csv'), index=True)

            print(f"PLS analysis on GRAIL results completed. Output saved in {ex_dir}.")
        else:
            save_figs = False
            print(f"PLS analysis on GRAIL results already exists in {ex_dir}.")

            # Load the results for plotting
            weighted_means_significant = pd.read_csv(os.path.join(ex_dir, 'weighted_means_significant.csv'))
            robust_features = json.load(open(os.path.join(ex_dir, 'robust_features.json')))
            eso_features = robust_features['escitalopram']
            psilo_features = robust_features['psilocybin']

        # Plot results ------------------------------------------------------------
        features_sorted = psilo_features + eso_features
        psilo_escit_cmap = custom_diverging_cmap(PSILO, ESCIT, n_colors=256)
        vmax = np.percentile(abs(weighted_means_significant[features_sorted].values), 75)
        barwidth = 0.65
        save_path = os.path.join(ex_dir, 'pls_grail_features.png')
        fig_size = (min(13, len(features_sorted)*barwidth), 5) # 13 is the max number of features to plot
        fig, ax = plot_diverging_bars(weighted_means_significant[features_sorted], 
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
    parser.add_argument('--metric', type=str, default='rho', help='Metric to use for the PLS analysis (rho or r)')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the output directory if it already exists')
    args = parser.parse_args()

    # At least one of the directories must be provided
    if args.grail_dir is None and args.attention_dir is None:
        raise ValueError("At least one of the directories must be provided.")

    # Run the main function
    main(args.grail_dir, args.attention_dir, args.output_dir_name, args.metric, args.overwrite)
