"""
Loads trained VGAEs and extracts the attention weights of the readout layer.
Analyses the attention weights in terms of resting state networks (RSNs) and
receptor distributions.

Attention weights are computed for all subjects with all fold models, then
they are compared across folds (correlation heatmap for each subject), and
post-hoc analyses are performed on the mean attention weights across folds.

Dependencies:
- data/raw/receptor_maps/f{atlas}/f{atlas}_receptor_maps.csv
- data/raw/receptor_maps/f{atlas}/same_rotation_for_each_receptor/f{receptor}.csv for each receptor

Author: Hanna Tolle
Date: 2025-05-03
License: BSD 3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 

import os
import torch
import torch.nn
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.files import add_project_root
from utils.configs import *
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.annotations import load_receptor_maps, load_rotated_rois
from utils.statsalg import get_fold_performance
from preprocessing.metrics import get_atlas, get_rsn_mapping
from utils.plotting import plot_brain_surface, COOLWARM, plot_stacked_percentages, plot_raincloud
from utils.statsalg import perform_dominance_analysis, perform_null_model_analysis


# Create experiment and logger -------------------------------------------------
ex = Experiment('attention_weights', ingredients=[data_ingredient, 
                                                  vgae_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'attention_weights'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Model weights directory, filenames and number of permutations
    weights_dir = os.path.join('outputs', 'graphTRIP', 'weights')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    
    # Manage log level/ verbosity
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE, MLP and dataset configs from weights_dir
    previous_config = load_ingredient_configs(weights_dir, ['vgae_model', 'mlp_model', 'dataset'])

    # Match configs of relevant ingredients
    ingredients = ['dataset', 'vgae_model']
    exceptions = []
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other compatibility checks
    num_folds = config_updates['dataset']['num_folds']
    weight_filenames = config_updates.get('weight_filenames', None)
    if weight_filenames is None:
        weight_filenames =  {'vgae': [f'k{k}_vgae_weights.pth' for k in range(num_folds)],
                             'test_fold_indices': ['test_fold_indices.csv']}
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    # VGAE must be a NodeLevelVGAE and pooling must be AttentionNetPooling
    assert config_updates['vgae_model']['model_type'] != 'GraphLevelVGAE', \
        "VGAE must be a NodeLevelVGAE or NodeLeveVGAE_Heterogeneous."
    
    # Pooling must be AttentionNetPooling or GlobalAttentionPooling
    assert config_updates['vgae_model']['pooling_cfg']['model_type'] == 'AttentionNetPooling' or \
           config_updates['vgae_model']['pooling_cfg']['model_type'] == 'GlobalAttentionPooling' or \
           config_updates['vgae_model']['pooling_cfg']['model_type'] == 'GraphTransformerPooling', \
        "Pooling must be AttentionNetPooling, GlobalAttentionPooling or GraphTransformerPooling."
    
    return config_updates

# Helper functions ------------------------------------------------------------

def get_attention_weights(vgaes, data, context_value=None):
    """
    Compute attention weights for each node in each subject using all VGAE models.
    
    Args:
        vgaes (list): List of trained VGAE models
        data (torch_geometric.data.Dataset): The full dataset
        context_value (float, optional): If provided, use this value instead of the original context
        
    Returns:
        pd.DataFrame: DataFrame with attention weights and atlas labels
    """    
    # Initialize arrays
    loader = DataLoader(data, batch_size=1, shuffle=False)
    num_subs = len(data)
    num_nodes = data[0].num_nodes
    num_folds = len(vgaes)
    attention_weights = np.zeros((num_folds, num_subs, num_nodes))

    # Compute attention weights for each subject and each fold
    for k, vgae in enumerate(vgaes):
        for sub, batch in enumerate(loader):
            # Get VGAE readout
            out = vgae(batch)
            if context_value is None:
                context = get_context(batch) # original context
            else:
                context = torch.full((out.z.shape[0], 1), context_value, dtype=torch.float) # artificial context
            z_with_context = torch.cat([out.mu, context], dim=1)
            
            # Get the attention weights
            attention_weights[k, sub] = vgae.pooling.get_attention_weights(z_with_context, batch.batch).detach().numpy().flatten()

    # Create DataFrame with attention weights using atlas labels
    atlas = get_atlas(data.atlas)
    decoded_labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in atlas['labels']]
    
    # Create a list of DataFrames, one for each fold
    attention_dfs = []
    for k in range(num_folds):
        df = pd.DataFrame(attention_weights[k], columns=decoded_labels)
        attention_dfs.append(df)
    
    return attention_dfs

def dominance_analysis_pipeline(regressors, attention_df, output_dir, analysis_name, rotated_roi_indices):
    """
    Perform dominance analysis pipeline on attention weights.
    
    Parameters:
    -----------
    regressors : pd.DataFrame
        DataFrame containing the regressors (receptors, RSNs, or combined)
    attention_df : pd.DataFrame
        DataFrame containing attention weights (num_subs, num_regions)
    output_dir : str
        Directory to save output files
    analysis_name : str
        Name of the analysis type (e.g., 'receptors', 'rsn', or 'combined')
    rotated_roi_indices : array-like
        Indices for rotated ROIs for null model analysis
    
    Returns:
    --------
    tuple : (da_stats_df, coefficients_df, results_df)
        da_stats_df : pd.DataFrame with dominance analysis statistics
        coefficients_df : pd.DataFrame with regression coefficients
        results_df : pd.DataFrame with R², t-stats, and p-values
    """
    # Get mean attention weights across subjects
    y = attention_df.mean(axis=0)
    
    # Perform dominance analysis
    da_stats = perform_dominance_analysis(regressors, y.values)
    da_stats['analysis'] = analysis_name

    # Plot stacked percentages
    save_path = os.path.join(output_dir, f'dominance_analysis_{analysis_name}.png')
    plot_stacked_percentages(df=da_stats, 
                           percentage_col='Percentage Relative Importance', 
                           save_path=save_path,
                           palette='mako',
                           figsize=(10, 3))

    # Significance testing
    real_rsquared, p_val, coef_df = perform_null_model_analysis(regressors, y, rotated_roi_indices)
    
    # Store results
    results = pd.DataFrame({
        'analysis': [analysis_name],
        'r2': [real_rsquared],
        'p_value': [p_val]
    })

    # Store coefficients
    coef_df['analysis'] = analysis_name

    return da_stats, coef_df, results

def perform_variance_analysis(attention_df_psilo: pd.DataFrame, attention_df_esci: pd.DataFrame):
    """
    Computes the drug-specific variance ratio for each brain region.

    Parameters:
    - attention_df_psilo: DataFrame of attention weights under psilocybin (subjects x regions)
    - attention_df_esci: DataFrame of attention weights under escitalopram (subjects x regions)

    Returns:
    - DataFrame with columns ['region', 'variance_ratio']
    """
    # Check that the two dataframes have the same shape and columns
    assert attention_df_psilo.shape == attention_df_esci.shape, "DataFrames must have the same shape."
    assert (attention_df_psilo.columns == attention_df_esci.columns).all(), "DataFrames must have the same columns."

    regions = attention_df_psilo.columns
    variance_ratios = []

    for region in regions:
        # Get attention weights for this region
        psilo_vals = attention_df_psilo[region].values
        esci_vals = attention_df_esci[region].values

        # Compute difference per patient
        diffs = psilo_vals - esci_vals

        # Variance of differences across patients
        var_d = diffs.var(ddof=1)  # Use sample variance (ddof=1)

        # Pool both conditions together
        pooled_vals = pd.concat([attention_df_psilo[region], attention_df_esci[region]])
        var_total = pooled_vals.var(ddof=1)

        # Avoid division by zero
        if var_total > 0:
            ratio = var_d / var_total
        else:
            ratio = float('nan')

        variance_ratios.append(ratio)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'region': regions,
        'variance_ratio': variance_ratios})

    return output_df

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    verbose = _config['verbose']
    output_dir = add_project_root(_config['output_dir'])
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    atlas = _config['dataset']['atlas']

    # Make output directory, get device and fix random seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    device = torch.device(_config['device'])

    # Load data and trained models
    data = load_data()
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device)
    vgaes = [vgae.eval() for vgae in vgaes]

    # Compute fold-wise performance of each model -----------------------------
    fold_performance = get_fold_performance(weights_dir)
    fold_performance.to_csv(os.path.join(output_dir, 'fold_performance.csv'), index=False)
    ex.add_artifact(os.path.join(output_dir, 'fold_performance.csv'))
    
    # Remaining attention analysis --------------------------------------------
    # Get context attributes if they exist
    context_attrs = _config['dataset'].get('context_attrs', [])
    context_attr_name = context_attrs[0] if len(context_attrs) > 0 else None
    if context_attr_name is not None:
        loader = DataLoader(data, batch_size=len(data), shuffle=False)
        batch = next(iter(loader))
        context_subs = batch.context_attr

    # Get the attention weights for the original context for all folds
    attention_dfs_original = get_attention_weights(vgaes, data)
    
    # Save attention weights for each fold
    for k, df in enumerate(attention_dfs_original):
        df.to_csv(os.path.join(output_dir, f'k{k}_attention_weights_original.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_attention_weights_original.csv'))

    # Compute mean attention weights across folds
    mean_attention_original = np.mean([df.values for df in attention_dfs_original], axis=0)
    mean_attention_df_original = pd.DataFrame(mean_attention_original, columns=attention_dfs_original[0].columns)
    mean_attention_df_original.to_csv(os.path.join(output_dir, 'mean_attention_weights_original.csv'), index=False)

    # Plot correlation heatmaps between folds for each subject
    num_subs = len(data)
    num_cols = 7
    num_rows = (num_subs + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows))
    axes = axes.flatten()
    corr_matrices = []

    for sub in range(num_subs):
        # Compute correlation matrix between folds for this subject
        sub_alignments = np.array([df.iloc[sub].values for df in attention_dfs_original])
        corr_matrix = np.corrcoef(sub_alignments)
        corr_matrices.append(corr_matrix)

        # Plot correlation heatmap
        ax = axes[sub]
        sns.heatmap(corr_matrix, ax=ax, cmap=COOLWARM, vmin=-1, vmax=1, 
                   square=True, annot=True, cbar=False)
        ax.set_title(f'Subject {sub}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove empty subplots
    for i in range(num_subs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fold_correlations.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    ex.add_artifact(save_path)

    # Save mean correlation matrix
    mean_corr_matrix = np.mean(corr_matrices, axis=0)
    np.savetxt(os.path.join(output_dir, 'mean_fold_correlations.csv'), mean_corr_matrix, delimiter=',')

    # Plot attention weights --------------------------------------------------
    # Plot the combined mean attention weights across all subjects
    vrange = (0, np.percentile(mean_attention_original.flatten(), 95))
    brain_data = mean_attention_df_original.mean().values
    plot_brain_surface(brain_data, 
                       atlas=atlas, 
                       threshold=None, 
                       cmap='mako', 
                       vrange=vrange, 
                       title='combined mean attention (original context)', 
                       save_path=os.path.join(output_dir, 'combined_mean_attention_original.png'));
    
    if not verbose:
        plt.close('all')

    # If context attributes exist, perform context-specific analyses
    if context_attr_name is not None:
        # Get the attention weights for each context value
        context_attr_values = np.unique(context_subs)
        mean_attention_dfs = []
        
        for value in context_attr_values:
            dfs = get_attention_weights(vgaes, data, value)
            # Compute mean attention weights across folds for this context value
            mean_attention = np.mean([df.values for df in dfs], axis=0)
            mean_df = pd.DataFrame(mean_attention, columns=dfs[0].columns)
            mean_attention_dfs.append(mean_df)
            
            # Save mean attention weights for this context value
            mean_df.to_csv(os.path.join(output_dir, f'mean_attention_weights_context_{value}.csv'), index=False)

        # If there are only 2 unique values, compute and save RSN-averaged differences
        if len(context_attr_values) == 2:
            # Get RSN mapping
            rsn_mapping, rsn_names = get_rsn_mapping(data.atlas)
            rsn_mapping = np.array(rsn_mapping)
            
            # Compute differences between contexts
            if context_attr_name == 'Condition':
                diff_df = mean_attention_dfs[1] - mean_attention_dfs[0] # psilo - escit
            else:
                diff_df = mean_attention_dfs[0] - mean_attention_dfs[1]
            
            # Average differences within each RSN
            rsn_diffs = {}
            for rsn_idx, rsn_name in enumerate(rsn_names):
                rsn_columns = diff_df.columns[rsn_mapping == rsn_idx]
                rsn_diffs[rsn_name] = diff_df[rsn_columns].mean(axis=1)
            
            # Save RSN-averaged differences
            pd.DataFrame(rsn_diffs).to_csv(os.path.join(output_dir, 'mean_rsn_attention_weights_context_diff.csv'), index=False)
            
            # Plot the difference on brain surface
            max_abs = np.percentile(np.abs(diff_df.values), 95)
            vrange = (-max_abs, max_abs)
            plot_brain_surface(diff_df.mean().values, 
                               atlas=atlas, 
                               threshold=None, 
                               cmap=COOLWARM, 
                               vrange=vrange, 
                               title=f'{context_attr_name} difference', 
                               save_path=os.path.join(output_dir, f'mean_attention_difference_{context_attr_name}.png'));

    # Receptor analysis --------------------------------------------------------
    print('Starting unimodal-transmodal axis + receptors analysis...')
    receptor_maps = load_receptor_maps(atlas=atlas)
    rotated_roi_indices = load_rotated_rois(data.atlas, 1000)

    ut_axis = load_ut_axis(atlas)
    X_combined = receptor_maps.copy()
    X_combined['ut_axis'] = ut_axis
    analysis_name = 'receptors_utaxis'
    da_stats, coef_df, results = dominance_analysis_pipeline(
        regressors=X_combined,
        attention_df=mean_attention_df_original,
        output_dir=output_dir,
        analysis_name=analysis_name,
        rotated_roi_indices=rotated_roi_indices)
    da_stats.to_csv(os.path.join(output_dir, f'da_{analysis_name}_stats.csv'))
    coef_df.to_csv(os.path.join(output_dir, f'da_{analysis_name}_coeffs.csv'))
    results.to_csv(os.path.join(output_dir, f'da_{analysis_name}_r2_pval_tstat.csv'), index=False)

    # RSN attention boxplots ---------------------------------------------------
    rsn_mapping, rsn_names = get_rsn_mapping(data.atlas)
    rsn_mapping = np.array(rsn_mapping)

    # Prepare data for plotting
    mean_rsn_attention = {rsn: [] for rsn in rsn_names}

    for rsn_idx, rsn_name in enumerate(rsn_names):
        # Get the columns for this RSN
        rsn_columns = mean_attention_df_original.columns[rsn_mapping == rsn_idx]

        # Average across RSN columns to get the mean attention for this RSN for each subject
        rsn_attention = mean_attention_df_original[rsn_columns].mean(axis=1)
        mean_rsn_attention[rsn_name] = list(rsn_attention)

    # Convert to DataFrame and save
    mean_rsn_attention = pd.DataFrame(mean_rsn_attention)
    mean_rsn_attention.to_csv(os.path.join(output_dir, 'mean_rsn_attention_weights_original.csv'), index=False)

    # RSN attention weights raincloud plot ------------------------------------
    # Plot the mean attention in each RSN for all subjects with the original context
    n_rsns = len(rsn_names)

    # Turn into dictionary
    mean_rsn_attention_dict = mean_rsn_attention.to_dict(orient='list')

    # Sort by mean attention
    mean_rsn_attention_dict = {k: v for k, v in sorted(mean_rsn_attention_dict.items(), key=lambda item: np.mean(item[1]))}

    # Plot the raincloud plot
    save_path = os.path.join(output_dir, 'mean_rsn_attention_raincloud_original.png')
    offset = 2
    colors = sns.color_palette("YlGnBu_r", n_rsns+offset)
    palette = {name: color for name, color in zip(rsn_names, colors)}
    plot_raincloud(mean_rsn_attention_dict, 
                palette=palette, 
                save_path=save_path, 
                alpha=0.7, 
                box_alpha=0.5,
                figsize=(5, 7),
                sort_by_mean=False)
    
    # Variance analysis --------------------------------------------------------
    # For each brain region, quantify how much of the total inter-subject
    # variance in attention is explained by the context attribute.
    if context_attr_name is not None and context_attr_name == 'Condition':
        # Get the attention weights for each context value
        attention_psilo = np.mean([df.values for df in get_attention_weights(vgaes, data, 1)], axis=0)
        attention_escit = np.mean([df.values for df in get_attention_weights(vgaes, data, -1)], axis=0)
        attention_psilo = pd.DataFrame(attention_psilo, columns=attention_dfs_original[0].columns)
        attention_escit = pd.DataFrame(attention_escit, columns=attention_dfs_original[0].columns)
        variance_df = perform_variance_analysis(attention_psilo, attention_escit)
        variance_df.to_csv(os.path.join(output_dir, 'variance_ratio.csv'), index=False)
        vmax = np.percentile(variance_df['variance_ratio'].values, 95)
        plot_brain_surface(variance_df['variance_ratio'].values, 
                            atlas=atlas, 
                            threshold=None, 
                            cmap='YlGnBu', 
                            vrange=(0, vmax), 
                            title=f'{context_attr_name} variance ratio', 
                            save_path=os.path.join(output_dir, f'variance_ratio_{context_attr_name}.png'));

        # Perform dominance analysis on the variance ratio
        print('Starting unimodal-transmodal axis + receptors analysis...')
        ut_axis = load_ut_axis(atlas)
        X_combined = receptor_maps.copy()
        X_combined['ut_axis'] = ut_axis
        y = variance_df['variance_ratio'].values
        analysis_name = 'variance_ratio'
        
        # Perform dominance analysis
        da_stats = perform_dominance_analysis(X_combined, y)
        da_stats['analysis'] = analysis_name

        # Plot stacked percentages
        save_path = os.path.join(output_dir, f'dominance_analysis_{analysis_name}.png')
        plot_stacked_percentages(df=da_stats, 
                            percentage_col='Percentage Relative Importance', 
                            save_path=save_path,
                            palette='mako',
                            figsize=(10, 3))

        # Significance testing
        real_rsquared, p_val, coef_df = perform_null_model_analysis(X_combined, y, rotated_roi_indices)
        
        # Store results
        coef_df['analysis'] = analysis_name
        results = pd.DataFrame({
            'analysis': [analysis_name],
            'r2': [real_rsquared],
            'p_value': [p_val]})

        # Save results
        results.to_csv(os.path.join(output_dir, f'dominance_analysis_{analysis_name}.csv'), index=False)
        coef_df.to_csv(os.path.join(output_dir, f'dominance_analysis_{analysis_name}_coeffs.csv'), index=False)
        da_stats.to_csv(os.path.join(output_dir, f'dominance_analysis_{analysis_name}_stats.csv'), index=False)

    plt.close('all')
    