"""
This scripts performs all post-hoc interpretability analyses.

Dependencies:
- experiments/outputs/graphtrip/weights/seed_*/
- experiments/outputs/graphtrip/grail/seed_*/
- experiments/outputs/graphtrip/regional_attributions/seed_*/
- same for medusa model

Outputs:
- experiments/outputs/grail/posthoc_analysis/
- experiments/outputs/regional_attributions/posthoc_analysis/
- experiments/outputs/test_biomarkers/

Author: Hanna M. Tolle
Date: 2025-12-19
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
from scipy.stats import norm, binom
from statsmodels.stats.multitest import fdrcorrection
from utils.files import add_project_root, rm_project_root
from utils.configs import load_configs_from_json, load_ingredient_configs
from utils.statsalg import compute_performance_weighted_means, ridge_cv_predict, get_fold_performance
from utils.annotations import load_receptor_maps, load_rotated_rois, load_ut_axis
from preprocessing.metrics import get_rsn_mapping
from utils.statsalg import calculate_consistency_scores
from experiments.run_experiment import run
from experiments.attention_weights import dominance_analysis_pipeline


# Helper functions -----------------------------------------------------------------------

def load_subject_dataframes(seed_dirs: list, filenames: list, inclusion_criteria: dict, 
                            excluded_subjects: list = None):
    """
    Load subject dataframes from GRAIL results organized in subject subdirectories.

    Args:
        base_dir (str): Base directory containing the seed directories with GRAIL results.
                        Example: 'outputs/graphtrip/grail/'
                        Base dir structure:
                        base_dir/
                            seed_0/
                                sub_0/
                                    k0_mean_alignments.csv
                                    k1_mean_alignments.csv
                                    ...
                                sub_1/
                                    ...
                            seed_1/
                                ...
        inclusion_criteria (dict): Criteria to include in the dataframes.
                                  Example: {'metric': 'rho', 'criterion': 'greater_than', 'threshold': 0.00}
        filenames (list): List of filenames to load, e.g., ['k0_mean_alignments.csv', 'k1_mean_alignments.csv', ...]
                         Rows will be ordered according to this list (and seed directories).
    
    Returns:
        all_subject_dfs (dict): Dictionary where keys are subject_IDs and values are DataFrames (num_models, num_biomarkers)
        performance (np.ndarray): Array containing the fold performances aligned with rows in subject dfs.
    """   
    # Extract fold numbers from filenames
    # Expected format: k{fold_num}_{rest}.csv
    filename_to_fold = {}
    for filename in filenames:
        try:
            fold_num = int(filename.split('_')[0][1:])  # Remove 'k' prefix
            filename_to_fold[filename] = fold_num
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract fold number from filename: {filename}. Expected format: k{{fold_num}}_{{rest}}.csv")
    
    # Compute fold_performances.csv for each seed_dir if it doesn't exist
    for seed_dir in seed_dirs:
        fold_perf_path = os.path.join(seed_dir, 'fold_performances.csv')
        if not os.path.exists(fold_perf_path):
            config_path = os.path.join(seed_dir, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {seed_dir}. Cannot compute fold_performances.")
            config = load_configs_from_json(config_path)
            mlp_weights_dir = add_project_root(rm_project_root(config['mlp_weights_dir']))
            fold_performances = get_fold_performance(mlp_weights_dir)
            fold_performances.to_csv(fold_perf_path, index=False)
    
    # Find all subject subdirectories from the first seed to determine number of subjects
    first_seed_dir = seed_dirs[0]
    subject_dirs = sorted([d for d in glob.glob(os.path.join(first_seed_dir, 'sub_*')) if os.path.isdir(d)])
    if not subject_dirs:
        raise ValueError(f"No subject subdirectories found in {first_seed_dir}")
    
    # Extract subject IDs
    subject_ids = []
    for sub_dir in subject_dirs:
        sub_id = int(os.path.basename(sub_dir).split('_')[1])
        subject_ids.append(sub_id)
    subject_ids = sorted(subject_ids)
    
    if excluded_subjects:
        subject_ids = [sub_id for sub_id in subject_ids if sub_id not in excluded_subjects]
        subject_dirs = [sub_dir for sub_dir in subject_dirs if int(os.path.basename(sub_dir).split('_')[1]) not in excluded_subjects]
    
    # First pass: identify which (seed, fold) combinations are missing for any subject
    # Structure: missing_models[seed_dir][fold_num] = True if missing for any subject
    missing_models = {}
    for seed_dir in seed_dirs:
        missing_models[seed_dir] = {}
        for filename in filenames:
            fold_num = filename_to_fold[filename]
            missing_models[seed_dir][fold_num] = False
            
            # Check if this file is missing for any subject
            for sub_id in subject_ids:
                sub_dir = os.path.join(seed_dir, f'sub_{sub_id}')
                file_path = os.path.join(sub_dir, filename)
                if not os.path.exists(file_path):
                    print(f"File {file_path} not found for subject {sub_id} in seed {seed_dir} and fold {fold_num}")
                    missing_models[seed_dir][fold_num] = True
                    break  # If missing for one subject, mark as missing
    
    # Load fold_performances and apply inclusion criteria
    # Structure: seed_fold_performances[seed_dir] = DataFrame with included folds
    seed_fold_performances = {}
    for seed_dir in seed_dirs:
        fold_perf_path = os.path.join(seed_dir, 'fold_performances.csv')
        fold_performance = pd.read_csv(fold_perf_path)
        fold_performance = fold_performance.sort_values('fold').reset_index(drop=True)
        
        # Apply inclusion criteria
        if inclusion_criteria['criterion'] == 'greater_than':
            mask = fold_performance[inclusion_criteria['metric']] > inclusion_criteria['threshold']
        elif inclusion_criteria['criterion'] == 'less_than':
            mask = fold_performance[inclusion_criteria['metric']] < inclusion_criteria['threshold']
        else:
            raise ValueError(f"Unknown criterion: {inclusion_criteria['criterion']}")
        
        included_fold_performance = fold_performance[mask].copy()
        included_fold_performance = included_fold_performance.sort_values('fold').reset_index(drop=True)
        seed_fold_performances[seed_dir] = included_fold_performance
    
    # Initialize output dictionary
    all_subject_dfs = {sub_id: [] for sub_id in subject_ids}
    all_fold_performances = []
    
    # Process each seed directory in order
    for seed_dir in seed_dirs:
        included_fold_performance = seed_fold_performances[seed_dir].copy()
        
        # Collect rows for all subjects for this seed
        seed_subject_rows = {sub_id: [] for sub_id in subject_ids}
        seed_performance_rows = []
        
        # Process each filename in order
        for filename in filenames:
            fold_num = filename_to_fold[filename]
            
            # Skip if this fold doesn't pass inclusion criteria
            if fold_num not in included_fold_performance['fold'].values:
                continue
            
            # Skip if this file is missing for any subject
            if missing_models[seed_dir][fold_num]:
                continue
            
            # Get performance row for this fold
            fold_perf_row = included_fold_performance[included_fold_performance['fold'] == fold_num]
            if len(fold_perf_row) == 0:
                continue
            
            # Load file for each subject
            all_subjects_have_file = True
            for sub_id in subject_ids:
                sub_dir = os.path.join(seed_dir, f'sub_{sub_id}')
                file_path = os.path.join(sub_dir, filename)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) == 1:
                        seed_subject_rows[sub_id].append(df.iloc[0])
                    else:
                        raise ValueError(f"Expected single row in {file_path}, found {len(df)}")
                else:
                    all_subjects_have_file = False
                    break
            
            # Only add performance row if all subjects have the file
            if all_subjects_have_file:
                seed_performance_rows.append(fold_perf_row.iloc[0])
        
        # Add rows to subject dataframes
        for sub_id in subject_ids:
            if seed_subject_rows[sub_id]:
                subject_df = pd.DataFrame(seed_subject_rows[sub_id])
                all_subject_dfs[sub_id].append(subject_df)
        
        # Add performance rows
        if seed_performance_rows:
            all_fold_performances.append(pd.DataFrame(seed_performance_rows))
    
    # Concatenate all subject dataframes (canonical order: seed_0/filenames[0], seed_0/filenames[1], ..., seed_N/filenames[-1])
    for sub_id in subject_ids:
        if all_subject_dfs[sub_id]:
            all_subject_dfs[sub_id] = pd.concat(all_subject_dfs[sub_id], axis=0, ignore_index=True)
        else:
            # If no data for this subject, create empty dataframe
            all_subject_dfs[sub_id] = pd.DataFrame()
    
    # Concatenate all fold_performances in the same canonical order
    if all_fold_performances:
        fold_performances_df = pd.concat(all_fold_performances, axis=0, ignore_index=True)
    else:
        fold_performances_df = pd.DataFrame()
    
    # Extract the performance metric as numpy array
    if len(fold_performances_df) > 0:
        performance = fold_performances_df[inclusion_criteria['metric']].values
    else:
        performance = np.array([])
    
    # Verify alignment: each subject df should have the same number of rows as performance array
    num_models = len(performance)
    for sub_id in subject_ids:
        if len(all_subject_dfs[sub_id]) != num_models:
            raise ValueError(f"Subject {sub_id} has {len(all_subject_dfs[sub_id])} rows but performance has {num_models} entries. "
                           f"Alignment mismatch!")
    
    return all_subject_dfs, performance

def aggregate_by_voting(df_significant: pd.DataFrame, alpha_per_run: float = 0.05):
    """
    df_significant: Boolean DataFrame (runs x features) where True = significant in that run.
    alpha_per_run: The FDR threshold used in the individual runs.
    """
    votes = df_significant.sum(axis=0)  # Number of 'True' per feature
    num_runs = len(df_significant)
    
    # Probability of getting 'votes' or more successes by chance
    # sf(k-1) gives P(X >= k)
    p_aggregated = binom.sf(votes - 1, num_runs, alpha_per_run)
    
    return pd.Series(p_aggregated, index=df_significant.columns)

def process_all_subjects(subject_data_dict):
    """
    Wrapper to process multiple subjects.
    
    Parameters:
        subject_data_dict (dict): Dictionary where keys are subject_IDs 
                                  and values are DataFrames (num_models, num_biomarkers)
                                  containing the signed z-scores.
                                  
    Returns:
        pd.DataFrame: aggregated p-values (num_subjects, num_biomarkers)
    """
    all_p_results = []
    for sub_id, df_runs in subject_data_dict.items():
        # Convert 0s to False and 1s to True
        df_runs = df_runs.astype(bool)
        results = aggregate_by_voting(df_runs, alpha_per_run=0.05)
        results.name = sub_id
        all_p_results.append(results)
    
    # Concatenate into final dfs
    final_p_df = pd.concat(all_p_results, axis=1).T
    return final_p_df

def assign_biomarker_categories(
    metric_dict, 
    alignment_dict, 
    threshold=0.05, 
    config_path='experiments/configs/biomarker_synergies.csv',
    criterion: str = 'less_than'
):
    """
    Categorizes biomarkers for each subject based on alignment signs and significance.

    Parameters:
        metric_dict (dict): Keys ['Shared', 'ITE', 'Escitalopram', 'Psilocybin'].
                            Values are DataFrames (num_subs, num_biomarkers) of metric values (e.g., FDR p-values or feature counts).
        alignment_dict (dict): Same keys. Values are DataFrames of signed mean alignments.
        threshold (float): Significance threshold.
        config_path (str): Path to biomarker synergies table.
        criterion (str): 'less_than' or 'greater_than'; how significance is determined in metric_dict.

    Returns:
        pd.DataFrame: (num_subs, num_biomarkers) containing category strings.
    """
    
    # 1. Load Configuration
    config_path = add_project_root(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config_df = pd.read_csv(config_path)
    model_cols = ['Shared', 'Psilocybin', 'Escitalopram', 'ITE']
    config_df[model_cols] = config_df[model_cols].astype(int)
    
    # 2. Discretise data:
    # 1 = positive align (resistance / better for escitalopram)
    # -1 = negative align (response / better for psilocybin)
    # 0 = not significant
    
    discretized_frames = {}
    for model in model_cols:
        if model not in metric_dict or model not in alignment_dict:
            raise KeyError(f"Missing model '{model}' in input dictionaries.")
            
        m_df = metric_dict[model]
        align_df = alignment_dict[model]
        
        # Create mask for significance based on criterion
        if criterion == "less_than":
            is_sig = (m_df < threshold).astype(int)
        elif criterion == "greater_than":
            is_sig = (m_df > threshold).astype(int)
        else:
            raise ValueError("criterion must be 'less_than' or 'greater_than'.")
        
        # Get sign of alignment (-1 or 1)
        signs = np.sign(align_df)
        
        # Set non-significant to 0
        discretized_frames[model] = (is_sig * signs).astype(int)

    # 3. Stack to make it look like:
    #                        Shared  Psilocybin  Escitalopram  ITE
    # Subject  Biomarker                                          
    # Sub_01   Biomarker_A       -1          -1            -1    0
    stacked_series = [df.stack().rename(name) for name, df in discretized_frames.items()]
    combined_long = pd.concat(stacked_series, axis=1)
    
    # 4. Merge with Config to get Categories
    merged_long = combined_long.reset_index().merge(
        config_df, 
        on=model_cols, 
        how='left')
    
    # 5. Handle uncategorized (invalid) combinations
    # e.g. if a combo isn't in the csv it gets assignef to'n.s.'
    merged_long['Biomarker_Category'] = merged_long['Biomarker_Category'].fillna('n.s.')
    
    # 6. Reshape back to (n_subs, n_biomarkers)
    idx_cols = combined_long.index.names
    if None in idx_cols: 
        idx_col_names = ['level_0', 'level_1']
    else:
        idx_col_names = idx_cols
        
    final_df = merged_long.pivot(
        index=idx_col_names[0], 
        columns=idx_col_names[1], 
        values='Biomarker_Category'
    )
    final_df.index.name = "Subject"
    final_df.columns.name = "Biomarker"
    
    return final_df

# Pipeline -----------------------------------------------------------------------------------

def analyse_grail_results(grail_dir=None, regional_attributions_dir=None, weights_dir=None, overwrite=False):

    # Add project root to paths
    if grail_dir:
        grail_dir = add_project_root(grail_dir)
    if weights_dir:
        weights_dir = add_project_root(weights_dir) 
    if regional_attributions_dir:
        regional_attributions_dir = add_project_root(regional_attributions_dir)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    inclusion_criteria = {'metric': 'rho', 
                          'criterion': 'greater_than', 
                          'threshold': 0.00} 

    # GRAIL posthoc --------------------------------------------------------------------------
    if grail_dir:
        ex_dir = os.path.join(grail_dir, 'posthoc_analysis')
        if not os.path.exists(ex_dir) or overwrite:

            os.makedirs(ex_dir, exist_ok=True)
            assert os.path.exists(grail_dir), f"GRAIL directory {grail_dir} not found"
            print(f"Running GRAIL posthoc analysis...")

            # Save posthoc analysis settings
            posthoc_config = {'inclusion_criteria': inclusion_criteria}
            with open(os.path.join(ex_dir, 'config.json'), 'w') as f:
                json.dump(posthoc_config, f, indent=4)

            # Load dataset config to infer num_folds
            dataset_config = load_ingredient_configs(os.path.join(grail_dir, 'seed_0'), ['dataset'])['dataset']
            num_folds = dataset_config['num_folds']

            # Load results from all seeds -----------------------------------------------------
            seed_dirs = sorted([d for d in glob.glob(os.path.join(grail_dir, 'seed_*')) if os.path.isdir(d)])
            if not seed_dirs:
                raise ValueError(f"No seed directories found in {grail_dir}")

            # Load subject GRAIL dfs and fold model performance
            excluded_subjects = None
            filenames = [f'k{k}_mean_alignments.csv' for k in range(num_folds)]
            all_subject_dfs, performance = load_subject_dataframes(seed_dirs, filenames=filenames,
                                                     inclusion_criteria=inclusion_criteria,
                                                     excluded_subjects=excluded_subjects)

            # Compute performance-weighted mean alignments ------------------------------------
            weighted_mean_alignments, _ = compute_performance_weighted_means(all_subject_dfs, performance) 
            weighted_mean_alignments.to_csv(os.path.join(ex_dir, 'weighted_mean_alignments.csv'), index=False)

            # Check agreement between fold models ---------------------------------------------
            agreement_scores = calculate_consistency_scores(all_subject_dfs)
            agreement_scores.to_csv(
                os.path.join(ex_dir, 'grail_agreement_scores.csv'), 
                header=["consistency_score"], 
                index=True,
                index_label="subject_id")

            # Aggregate results from permutation tests ----------------------------------------
            # Load the dfs that indicate which biomarkers were significant in each fold
            filenames = [f'k{k}_selected_features.csv' for k in range(num_folds)]
            all_subject_dfs, _ = load_subject_dataframes(seed_dirs, filenames=filenames,
                                                     inclusion_criteria=inclusion_criteria,
                                                     excluded_subjects=excluded_subjects)

            # Aggregate p-values from different models for each subject
            alignment_pvalues = process_all_subjects(all_subject_dfs)

            # Perform FDR-correction on each row in alignment_pvalues
            def get_fdr(row):
                return fdrcorrection(row, alpha=0.05)[1]
            alignment_pvalues_fdr = alignment_pvalues.apply(get_fdr, axis=1, result_type='broadcast')
            alignment_pvalues_fdr.to_csv(os.path.join(ex_dir, 'fdr_corrected_alignment_pvalues.csv'), index=False)

            # Also compute the proportion of runs that were significant for each biomarker
            feature_counts = []
            all_sub_ids = list(all_subject_dfs.keys())
            for df in all_subject_dfs.values():
                num_rows = len(df)
                feature_counts.append((df.sum(axis=0) / num_rows).to_dict())
            feature_counts = pd.DataFrame(feature_counts, index=all_sub_ids)
            feature_counts.to_csv(os.path.join(ex_dir, 'feature_counts.csv'), index=False)

    # Regional attributions posthoc analysis ---------------------------------------------------
    if regional_attributions_dir:
        ex_dir = os.path.join(regional_attributions_dir, 'posthoc_analysis')
        if not os.path.exists(ex_dir) or overwrite:

            os.makedirs(ex_dir, exist_ok=True)
            assert os.path.exists(regional_attributions_dir), f"Regional attributions directory {regional_attributions_dir} not found"
            print(f"Running regional importance analysis...")

            # Save posthoc analysis settings
            posthoc_config = {'inclusion_criteria': inclusion_criteria}
            with open(os.path.join(ex_dir, 'config.json'), 'w') as f:
                json.dump(posthoc_config, f, indent=4)

            # Load dataset config to infer num_folds
            dataset_config = load_ingredient_configs(os.path.join(regional_attributions_dir, 'seed_0'), ['dataset'])['dataset']
            num_folds = dataset_config['num_folds']

            # Load results from all seeds -----------------------------------------------------
            seed_dirs = sorted([d for d in glob.glob(os.path.join(regional_attributions_dir, 'seed_*')) if os.path.isdir(d)])
            if not seed_dirs:
                raise ValueError(f"No seed directories found in {regional_attributions_dir}")

            excluded_subjects = None
            filenames = [f'k{k}_regional_attributions.csv' for k in range(num_folds)]
            all_subject_dfs, performance = load_subject_dataframes(seed_dirs, filenames=filenames,
                                                     inclusion_criteria=inclusion_criteria,
                                                     excluded_subjects=excluded_subjects)

            # Compute performance-weighted attributions --------------------------------------
            weighted_mean_attributions, _ = compute_performance_weighted_means(all_subject_dfs, performance) 
            weighted_mean_attributions.to_csv(os.path.join(ex_dir, 'weighted_mean_attributions.csv'), index=False)

            # Check agreement between fold models --------------------------------------------
            agreement_scores = calculate_consistency_scores(all_subject_dfs)
            agreement_scores.to_csv(
                os.path.join(ex_dir, 'regional_attributions_agreement_scores.csv'), 
                header=["consistency_score"], 
                index=True,
                index_label="subject_id")

            # Perform dominance analysis on weighted-mean attention weights ------------------
            atlas = 'schaefer100'
            receptor_maps = load_receptor_maps(atlas=atlas)
            rotated_roi_indices = load_rotated_rois(atlas, 1000)
            ut_axis = load_ut_axis(atlas)
            X_combined = receptor_maps.copy()
            X_combined['ut_axis'] = ut_axis
            analysis_name = 'receptors_utaxis'
            da_stats, coef_df, results = dominance_analysis_pipeline(
                regressors=X_combined,
                attention_df=weighted_mean_attributions,
                output_dir=ex_dir,
                analysis_name=analysis_name,
                rotated_roi_indices=rotated_roi_indices)
            da_stats.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_stats.csv'))
            coef_df.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_coeffs.csv'), index=False)
            results.to_csv(os.path.join(ex_dir, f'da_{analysis_name}_r2_pval_tstat.csv'), index=False)

            # Compute average weighted-mean attention weights per RSN --------------------------
            rsn_mapping, rsn_names = get_rsn_mapping(atlas)
            rsn_mapping = np.array(rsn_mapping)
            mean_rsn_grad_weights = {rsn: [] for rsn in rsn_names}
            for rsn_idx, rsn_name in enumerate(rsn_names):
                rsn_columns = weighted_mean_attributions.columns[rsn_mapping == rsn_idx]
                rsn_grad_weights = weighted_mean_attributions[rsn_columns].mean(axis=1)
                mean_rsn_grad_weights[rsn_name] = list(rsn_grad_weights)
            mean_rsn_grad_weights = pd.DataFrame(mean_rsn_grad_weights)
            mean_rsn_grad_weights.to_csv(os.path.join(ex_dir, 'weighted_mean_rsn_attributions.csv'), index=False)

            print(f"Regional attributions posthoc analysis completed. Output saved in {ex_dir}.")
        else:
            print(f"Regional attributions posthoc experiment already exists in {ex_dir}.")

    # Test biomarkers ---------------------------------------------------------------------------
    if weights_dir:
        weights_dirs = sorted([d for d in glob.glob(os.path.join(weights_dir, 'seed_*'))
                               if os.path.isdir(d)])
        exname = 'test_biomarkers'
        ex_dir = os.path.join(weights_dir, '..', 'test_biomarkers')
        if not os.path.exists(ex_dir) or overwrite:
            config_updates = {}
            config_updates['output_dir'] = ex_dir
            config_updates['weights_dir'] = weights_dirs[0]
            config_updates['seed'] = 0
            config_updates['verbose'] = True
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

# Main ---------------------------------------------------------------------------------------------
def main(core_model_dir, medusa_model_dir, overwrite=False):
    core_model_dir = add_project_root(core_model_dir)
    medusa_model_dir = add_project_root(medusa_model_dir)

    # Perform post-hoc analysis for each model -----------------------------------------------------
    grail_dirs = {
        'Shared': os.path.join(core_model_dir, 'grail'),                      # Shared biomarkers
        'ITE': os.path.join(medusa_model_dir, 'medusa_grail'),                # ITE biomarkers (negative = P response, positive = E response)
        'Escitalopram': os.path.join(medusa_model_dir, 'grail_escitalopram'), # Escitalopram biomarkers
        'Psilocybin': os.path.join(medusa_model_dir, 'grail_psilocybin'),     # Psilocybin biomarkers
    }

    # graphTRIP core model
    grail_dir = grail_dirs['Shared']
    regional_attributions_dir = os.path.join(core_model_dir, 'regional_attributions')
    weights_dir = os.path.join(core_model_dir, 'weights')
    analyse_grail_results(grail_dir, regional_attributions_dir, weights_dir, overwrite=overwrite)

    # graphTRIP medusa model
    medusa_grail_dir = grail_dirs['ITE']
    regional_attributions_dir = os.path.join(medusa_model_dir, 'regional_attributions')
    analyse_grail_results(medusa_grail_dir, regional_attributions_dir, overwrite=overwrite)

    # graphTRIP medusa model, escitalopram mode
    medusa_grail_dir = grail_dirs['Escitalopram']
    analyse_grail_results(medusa_grail_dir, overwrite=overwrite)

    # graphTRIP medusa model, psilocybin mode
    medusa_grail_dir = grail_dirs['Psilocybin']
    analyse_grail_results(medusa_grail_dir, overwrite=overwrite)

    # Post-posthoc analysis ------------------------------------------------------------------------
    # Load FDR-corrected alignment p-values and weighted-mean alignments
    all_fdrs = {}
    all_alignments = {}
    for model_name, grail_dir in grail_dirs.items():
        posthoc_dir = os.path.join(grail_dir, 'posthoc_analysis')
        alignment_fdrs = pd.read_csv(os.path.join(posthoc_dir, 'fdr_corrected_alignment_pvalues.csv'))
        all_fdrs[model_name] = alignment_fdrs
        alignment_weights = pd.read_csv(os.path.join(posthoc_dir, 'weighted_mean_alignments.csv')) 

    # Assign biomarkers to categories
    output_dir = os.path.join('outputs', 'biomarker_categories')
    output_dir = add_project_root(output_dir)
    if not os.path.exists(output_dir) or overwrite:
        os.makedirs(output_dir, exist_ok=True)
        biomarker_categories = assign_biomarker_categories(all_fdrs, all_alignments, threshold=0.05, criterion='less_than')
        biomarker_categories.to_csv(os.path.join(output_dir, 'biomarker_categories.csv'), index=False)
    else:
        print(f"Biomarker categories already exist in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--core_model_dir', type=str, default='outputs/graphtrip/')
    parser.add_argument('--medusa_model_dir', type=str, default='outputs/x_graphtrip/')
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args.core_model_dir, args.medusa_model_dir, args.overwrite)