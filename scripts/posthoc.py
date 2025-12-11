"""
This scripts performs all post-hoc analyses for graphTRIP.

Dependencies:
- experiments/outputs/weights/seed_*/
- experiments/outputs/graphtrip/grail/seed_*/

Outputs:
- outputs/graphtrip/grail/posthoc_analysis/
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
import argparse
import glob
import json
import pandas as pd
import numpy as np
from utils.files import add_project_root
from utils.configs import load_configs_from_json
from utils.statsalg import load_fold_data, compute_performance_weighted_means, ridge_cv_predict
from utils.annotations import load_receptor_maps, load_rotated_rois, load_ut_axis
from preprocessing.metrics import get_rsn_mapping
from utils.statsalg import calculate_consistency_scores
from experiments.run_experiment import run
from experiments.attention_weights import dominance_analysis_pipeline


def main(grail_dir=None, weights_dir=None, verbose=False, seed=0):

    # Add project root to paths
    if grail_dir:
        grail_dir = add_project_root(grail_dir)
    if weights_dir:
        weights_dir = add_project_root(weights_dir) 
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config_updates = {}
    config_updates['verbose'] = verbose
    config_updates['seed'] = seed

    # GRAIL posthoc ----------------------------------------------------------
    if grail_dir:

        ex_dir = os.path.join(grail_dir, 'posthoc_analysis')
        if not os.path.exists(ex_dir):
            os.makedirs(ex_dir, exist_ok=True)
            assert os.path.exists(grail_dir), f"GRAIL directory {grail_dir} not found"
            
            # Post-hoc analysis settings -------------------------------------------------
            # CV-fold model inclusion criteria (true-vs-pred correlation > 0)
            inclusion_criteria = {'metric': 'rho', 
                                  'criterion': 'greater_than', 
                                  'threshold': 0.00} 
            
            # Filtering criteria for PLS analysis
            filtering_criteria = {'effect_size': 'very large',
                                  'fdr_p_value': 0.05,
                                  'percentile': 25}

            # Save all analysis settings
            posthoc_config = {'inclusion_criteria': inclusion_criteria,
                              'filtering_criteria': filtering_criteria}
            with open(os.path.join(ex_dir, 'config.json'), 'w') as f:
                json.dump(posthoc_config, f, indent=4)

            # Load the config from grail_dir
            config_file = os.path.join(grail_dir, 'seed_0', 'config.json')
            grail_config = load_configs_from_json(config_file)
            medusa = grail_config.get('medusa', False) # Apply two-sided filtering for medusa models

            # -----------------------------------------------------------------------------
            # GRADIENT ALIGNMENT ANALYSIS
            # -----------------------------------------------------------------------------
            print(f"Running gradient alignment analysis...")

            # Load results from all seeds -------------------------------------------------
            seed_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'seed_*')) if os.path.isdir(d)])
            if not seed_dirs:
                raise ValueError(f"No seed directories found in {grail_dir}")
            filenames = sorted([os.path.basename(f) for f in glob.glob(os.path.join(seed_dirs[0], 'k*_mean_alignments.csv'))
                                if os.path.isfile(f)])

            # Load subject GRAIL dfs and fold model performance
            all_subject_dfs, fold_performances = load_fold_data(seed_dirs, filenames, inclusion_criteria)
            performance = fold_performances[inclusion_criteria['metric']].values

            # Compute performance-weighted mean alignments --------------------------------
            weighted_mean_alignments, weighted_mean_alignments_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance) 
            weighted_mean_alignments_stats.to_csv(os.path.join(ex_dir, 'weighted_mean_alignments_stats.csv'))
            weighted_mean_alignments.to_csv(os.path.join(ex_dir, 'weighted_mean_alignments.csv'), index=False)

            # Check agreement between fold models ------------------------------------------
            agreement_scores = calculate_consistency_scores(all_subject_dfs)
            agreement_scores.to_csv(
                os.path.join(ex_dir, 'grail_agreement_scores.csv'), 
                header=["consistency_score"], 
                index=True,
                index_label="subject_id")

            # Filter features -------------------------------------------------------------
            # 1. Filter based on significance and effect size
            filtered_features = list(
                weighted_mean_alignments_stats[
                    (weighted_mean_alignments_stats['fdr_p_value'] < filtering_criteria['fdr_p_value']) &
                    (weighted_mean_alignments_stats['effect_size'] == filtering_criteria['effect_size'])
                ].index)

            # 2. Filter based on percentile of mean alignment magnitude
            if medusa:
                half_pctl = filtering_criteria['percentile'] / 2  
                
                # Positive features
                pos_indices = weighted_mean_alignments_stats[weighted_mean_alignments_stats['t_statistic'] > 0].index
                pos_means = weighted_mean_alignments[pos_indices].mean() 
                pos_threshold = np.percentile(pos_means, 100 - half_pctl)
                filtered_positive = pos_means[pos_means > pos_threshold].index.tolist()

                # Negative features
                neg_indices = weighted_mean_alignments_stats[weighted_mean_alignments_stats['t_statistic'] < 0].index
                neg_means = weighted_mean_alignments[neg_indices].mean()
                neg_threshold = np.percentile(np.abs(neg_means), 100 - half_pctl)
                filtered_negative = neg_means[np.abs(neg_means) > neg_threshold].index.tolist()

                # Combine and filter based on intersection
                both = filtered_positive + filtered_negative
                filtered_features = [f for f in filtered_features if f in both]
            else:
                pctl = filtering_criteria['percentile']
                means = weighted_mean_alignments.mean()
                threshold = np.percentile(np.abs(means), 100 - pctl)
                filtered_features = [col for col in filtered_features if np.abs(means[col]) > threshold]
            
            # Save results
            with open(os.path.join(ex_dir, 'filtered_features.json'), 'w') as f:
                json.dump(filtered_features, f, indent=4)

            # -----------------------------------------------------------------------------
            # REGIONAL IMPORTANCE ANALYSIS
            # -----------------------------------------------------------------------------
            print(f"Running regional importance analysis...")

            # Load results from all seeds -------------------------------------------------
            filenames = sorted([os.path.basename(f) for f in glob.glob(os.path.join(seed_dirs[0], 'k*_regional_grad_weights.csv'))
                                if os.path.isfile(f)])

            # Load subject regional gradient weights dfs and fold model performance
            all_subject_dfs, fold_performances = load_fold_data(seed_dirs, filenames, inclusion_criteria)
            performance = fold_performances[inclusion_criteria['metric']].values

            # Compute performance-weighted regional gradient weights ----------------------
            wm_grad_weights, wm_grad_weights_stats = \
                compute_performance_weighted_means(all_subject_dfs, performance)
            wm_grad_weights_stats.to_csv(os.path.join(ex_dir, 'weighted_mean_grad_weights_stats.csv'))
            wm_grad_weights.to_csv(os.path.join(ex_dir, 'weighted_mean_grad_weights.csv'), index=False)

            # Check agreement between fold models ------------------------------------------
            agreement_scores = calculate_consistency_scores(all_subject_dfs)
            agreement_scores.to_csv(
                os.path.join(ex_dir, 'regional_importance_agreement_scores.csv'), 
                header=["consistency_score"], 
                index=True,
                index_label="subject_id")

            # Perform dominance analysis on weighted-mean attention weights ---------------
            atlas = 'schaefer100'
            receptor_maps = load_receptor_maps(atlas=atlas)
            rotated_roi_indices = load_rotated_rois(atlas, 1000)
            ut_axis = load_ut_axis(atlas)
            X_combined = receptor_maps.copy()
            X_combined['ut_axis'] = ut_axis
            analysis_name = 'receptors_utaxis'
            da_stats, coef_df, results = dominance_analysis_pipeline(
                regressors=X_combined,
                attention_df=wm_grad_weights,
                output_dir=ex_dir,
                analysis_name=analysis_name,
                rotated_roi_indices=rotated_roi_indices)
            da_stats.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_stats.csv'))
            coef_df.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_coeffs.csv'), index=False)
            results.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_r2_pval_tstat.csv'), index=False)

            # Compute average weighted-mean attention weights per RSN --------------------------------------
            rsn_mapping, rsn_names = get_rsn_mapping(atlas)
            rsn_mapping = np.array(rsn_mapping)
            mean_rsn_grad_weights = {rsn: [] for rsn in rsn_names}
            for rsn_idx, rsn_name in enumerate(rsn_names):
                rsn_columns = wm_grad_weights.columns[rsn_mapping == rsn_idx]
                rsn_grad_weights = wm_grad_weights[rsn_columns].mean(axis=1)
                mean_rsn_grad_weights[rsn_name] = list(rsn_grad_weights)
            mean_rsn_grad_weights = pd.DataFrame(mean_rsn_grad_weights)
            mean_rsn_grad_weights.to_csv(os.path.join(ex_dir, 'weighted_mean_rsn_grad_weights.csv'), index=False)

            print(f"GRAIL posthoc analysis completed. Output saved in {ex_dir}.")
        else:
            print(f"GRAIL posthoc experiment already exists in {ex_dir}.")

    # Test biomarkers ----------------------------------------------------------
    if weights_dir:
        weights_dirs = sorted([d for d in glob.glob(os.path.join(weights_dir, 'seed_*'))
                               if os.path.isdir(d)])
        config = load_configs_from_json(os.path.join(weights_dirs[0], 'config.json'))        
        exname = 'test_biomarkers'
        ex_dir = os.path.join(weights_dir, '..', 'test_biomarkers')
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
            df = ridge_cv_predict(X, y, subject_ids=feature_values['sub'].values, n_splits=7, random_state=0)
            df.to_csv(os.path.join(ex_dir, 'ridge_cv_predict.csv'), index=False)
        else:
            print(f"Biomarkers experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python posthoc.py --grail_dir outputs/graphtrip/grail -s 0 -v
    python posthoc.py --weights_dir outputs/graphtrip/weights -s 0 -v
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--grail_dir', type=str, default=None,
                        help='Path to the GRAIL directory (for GRAIL posthoc analysis)')
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='Path to the weights directory (for test_biomarkers analysis)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Run the main function
    main(args.grail_dir, args.weights_dir, args.verbose, args.seed)