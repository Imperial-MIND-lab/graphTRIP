"""
Util functions for statistical analysis and algorithms.

Author: Hanna M. Tolle
Date: 2025-02-12
License: BSD-3-Clause
"""

import os
from typing import List, Dict
from dominance_analysis import Dominance
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score, r2_score
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV


# Helper functions --------------------------------------------------------------

def calculate_cohens_d(data1, data2=None):
    """Calculate Cohen's d effect size."""
    if data2 is None:  # One-sample case
        d = np.mean(data1) / np.std(data1, ddof=1)
    else:  # Two-sample case
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(data1) - np.mean(data2)) / pooled_se
    return d

def min_significant_r(n, alpha=0.05):
    """
    Calculate the minimum significant r value for a given sample size and significance level.
    
    Parameters:
    -----------
    n (int): Sample size
    alpha (float): Significance level (default: 0.05)
    
    Returns:
    --------
    r_min (float): Minimum significant r value
    """
    # Get the critical t-value for a two-tailed test
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 2)
    
    # Compute the minimum significant r
    r_min = t_critical / (t_critical**2 + (n - 2))**0.5
    return r_min

# Dominance analysis ------------------------------------------------------------

def perform_dominance_analysis(X, y):
    """
    Perform dominance analysis without modifying the input DataFrame.
    
    Parameters:
    -----------
    X (pd.DataFrame): Input features
    y (pd.Series or np.array): Target variable
    
    Returns:
    --------
    dict: Dominance statistics
    """
    # Create a copy of X to avoid modifying the original
    X_copy = X.copy()
    X_copy['target'] = y
    dominance_regression = Dominance(data=X_copy, target='target', objective=1)
    incr_variable_rsquare = dominance_regression.incremental_rsquare()
    return dominance_regression.dominance_stats()

def perform_null_model_analysis(true_x, y, rotated_roi_indices, make_plot=False):
    """
    Perform null model analysis to compare the R-squared of the actual model with the R-squared of the null model.
    Also performs individual linear regressions for each regressor.
    
    Parameters:
    -----------
    true_x (pd.DataFrame): DataFrame containing the true regressor values (num_samples, num_regressors)
    y (np.ndarray): Array containing the target values (num_samples,)
    rotated_roi_indices (np.ndarray): Array of shape (num_rois, n_permutations) containing permuted ROI indices
                                     that preserve spatial autocorrelation
    make_plot (bool): If True, creates a histogram of the null distribution with the real R² marked

    Returns:
    --------
    real_rsquared (float): R-squared of the actual model
    p_val (float): P-value from permutation test
    coef_df (pd.DataFrame): DataFrame with regressor names and their coefficients from individual regressions
    """
    n_permutations = rotated_roi_indices.shape[1]
    
    # Compute null distribution of R² using rotated ROIs
    null_rsquared = []
    for perm in range(n_permutations):
        # Apply the permutation to each regressor
        rotated_x = pd.DataFrame(
            {col: true_x[col].values[rotated_roi_indices[:, perm]] for col in true_x.columns},
            columns=true_x.columns)
        
        # Fit model and get R² score
        model = LinearRegression()
        model.fit(rotated_x, y)
        null_rsquared.append(model.score(rotated_x, y))

    # Full model for R-squared
    model = LinearRegression()
    model.fit(true_x, y)
    real_rsquared = model.score(true_x, y)

    # Individual regressions for each regressor
    coefficients = []
    for regressor in true_x.columns:
        X_single = true_x[regressor].values.reshape(-1, 1)
        model_single = LinearRegression()
        model_single.fit(X_single, y)
        coefficients.append(model_single.coef_[0])

    # Create DataFrame with coefficients from individual regressions
    coef_df = pd.DataFrame({
        'regressor': true_x.columns,
        'coefficient': coefficients
    })

    # Compare real R² against null distribution
    # Calculate p-value directly from permutation test
    p_val = (np.sum(null_rsquared >= real_rsquared) + 1) / (len(null_rsquared) + 1)
    
    if make_plot:        
        plt.figure(figsize=(10, 6))
        plt.hist(null_rsquared, bins=30, alpha=0.5, color='blue', density=True)
        # Format p-value in scientific notation if it's very small
        p_val_str = f'{p_val:.3e}' if p_val < 0.001 else f'{p_val:.3f}'
        plt.axvline(x=real_rsquared, color='red', linestyle='-', linewidth=2, 
                   label=f'Real R² = {real_rsquared:.3f}\np = {p_val_str}')
        plt.xlabel('R²')
        plt.ylabel('Density')
        plt.title('Null Distribution of R² Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return real_rsquared, p_val, coef_df

# Clustering algorithms ---------------------------------------------------------

def louvain_consensus_clustering(corr_data, num_seeds, seed, 
                                 threshold=None, cmap='plasma', save_path=None):
    """
    Analyze a correlation matrix using community detection with multiple seeds.
    
    Parameters:
    -----------
    corr_data (numpy.ndarray): Correlation matrix to analyze (num_samples, num_samples)
    num_seeds (int): Number of seeds to use for community detection (default: 10).
    seed (int): Initial seed for reproducibility (default: 291).
    threshold (float): Optional threshold for co-association matrix (default: None).
    save_path (str): Path to save the figure (default: None).
    
    Returns:
    --------
    communities (list): Final aggregated community assignments.
    sort_idx (numpy.ndarray): Indices for sorting by community.
    coassoc_matrix (numpy.ndarray): Co-association matrix.
    """
    num_samples = corr_data.shape[0]
    G = nx.from_numpy_array(corr_data)

    # Run Louvain algorithm multiple times
    all_communities = []
    for i in range(num_seeds):
        communities = nx.community.louvain_communities(G, seed=seed+i)
        community_assignment = np.zeros(num_samples, dtype=int)
        for idx, community in enumerate(communities):
            for node in community:
                community_assignment[node] = idx
        all_communities.append(community_assignment)

    # Create co-association matrix
    coassoc_matrix = np.zeros((num_samples, num_samples), dtype=float)
    for assignment in all_communities:
        for i in range(num_samples):
            for j in range(num_samples):
                if assignment[i] == assignment[j]:
                    coassoc_matrix[i, j] += 1
    coassoc_matrix /= num_seeds

    # Threshold co-association matrix if a threshold is provided
    if threshold is not None:
        coassoc_matrix = (coassoc_matrix >= threshold).astype(float)

    # Recompute communities based on co-association matrix
    G_coassoc = nx.from_numpy_array(coassoc_matrix)
    final_communities = nx.community.louvain_communities(G_coassoc, seed=seed)

    # Get labels
    labels = np.zeros(num_samples, dtype=int)
    for idx, community in enumerate(final_communities):
        for node in community:
            labels[node] = idx
    sort_idx = np.argsort(labels)
    coassoc_sorted = coassoc_matrix[sort_idx, :][:, sort_idx]

    # Plotting ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Heatmap for correlation matrix
    corr_data_sorted = corr_data[sort_idx, :][:, sort_idx]

    # Get all upper triangular elements to set vmax
    triu_idx = np.triu_indices(num_samples, k=1)
    triu_corrs = corr_data[triu_idx[0], triu_idx[1]]
    vmax = np.percentile(abs(triu_corrs), 95)

    sns.heatmap(corr_data_sorted, annot=False, cmap=cmap, square=True,
                xticklabels=False, yticklabels=False, ax=ax1, vmax=vmax)
    ax1.set_title('Correlation Matrix')

    # Heatmap for sorted and thresholded co-association matrix
    sns.heatmap(coassoc_sorted, annot=False, cmap=cmap, square=True,
                xticklabels=False, yticklabels=False, ax=ax2)
    ax2.set_title('Co-association Matrix')

    # Add labels to show community boundaries
    tick_positions = []
    tick_labels = []
    current_pos = 0
    for i, community in enumerate(final_communities):
        tick_positions.append(current_pos + len(community) // 2)
        tick_labels.append(f"C{i+1}")
        current_pos += len(community)
    ax2.set_xticks(tick_positions)
    ax2.set_yticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, fontsize=10)
    ax2.set_yticklabels(tick_labels, fontsize=10)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    
    return labels, coassoc_matrix

def compute_pca(X, max_components=None, conditions=None, standardise=True, save_path=None):
    """
    Performs PCA on the input data X and produces some plots:
     - the data in the space of the first two principal components
     - the explained variance of all computed principal components (with plt.grid())
    
    Parameters:
    ----------
     X (2D numpy array): The data to perform PCA on (num_samples, num_features).
     max_components (int): The maximum number of principal components to compute.
     feature_names (list): The names of the features.
     conditions (numpy array): The class labels for each sample (num_samples,).
        If provided, the data points are colored according to their class.
        Each unique class gets a different color.
    save_path (str): Path to save the figure.

    Returns:
    --------
     pca (sklearn PCA object): The PCA object.
     X_pca (2D numpy array): The data in the space of the first two principal components.
    """
    # Initialize PCA
    if max_components is None:
        max_components = min(X.shape)
    pca = PCA(n_components=max_components)

    # Standardise the data if requested
    if standardise:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X

    # Fit and transform the data
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 2: Data in PC space
    if conditions is not None:
        # Get unique classes and assign colors
        unique_classes = np.unique(conditions)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        # Plot each class with a different color
        for class_label, color in zip(unique_classes, colors):
            mask = conditions == class_label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[color], label=str(class_label))
        
        ax1.legend()
    else:
        ax1.scatter(X_pca[:, 0], X_pca[:, 1])
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Data in PC Space')
    ax1.grid(True)
    
    # Plot 3: Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    ax2.plot(range(1, len(explained_var) + 1), explained_var, 'bo-', label='Individual')
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Scree Plot')
    ax2.grid(True)
    ax2.legend()
    
    # Set x-ticks to integers only
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    
    return pca, scaler

def test_cluster_testfold_independence(labels, testfold_indices, n_permutations=10000):
    """
    Test if testfold assignments are predictive of cluster assignments using
    chi-square test of independence, mutual information, and permutation testing.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Cluster assignments for each subject (shape: n_subjects,)
    testfold_indices : numpy.ndarray
        Test fold assignments for each subject (shape: n_subjects,)
    n_permutations : int
        Number of permutations for the test
    
    Returns:
    --------
    dict
        Contains original chi-square statistic, p-values, mutual information,
        and contingency table
    """
    
    # Compute original contingency table and chi-square statistic
    unique_clusters = np.unique(labels)
    unique_folds = np.unique(testfold_indices)
    orig_contingency = np.zeros((len(unique_clusters), len(unique_folds)))
    
    for i, cluster in enumerate(unique_clusters):
        for j, fold in enumerate(unique_folds):
            orig_contingency[i, j] = np.sum((labels == cluster) & (testfold_indices == fold))
            
    chi2_stat, p_value, _, _ = chi2_contingency(orig_contingency)
    
    # Compute mutual information
    mi = mutual_info_score(labels, testfold_indices)
    
    # Perform permutation test
    perm_chi2_stats = np.zeros(n_permutations)
    perm_mi_scores = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Permute the cluster labels
        perm_labels = np.random.permutation(labels)
        
        # Compute contingency table for permuted data
        perm_contingency = np.zeros((len(unique_clusters), len(unique_folds)))
        for j, cluster in enumerate(unique_clusters):
            for k, fold in enumerate(unique_folds):
                perm_contingency[j, k] = np.sum((perm_labels == cluster) & (testfold_indices == fold))
        
        # Compute chi-square statistic for permuted data
        perm_chi2_stats[i], _, _, _ = chi2_contingency(perm_contingency)
        
        # Compute mutual information for permuted data
        perm_mi_scores[i] = mutual_info_score(perm_labels, testfold_indices)
    
    # Calculate permutation p-values
    perm_p_value_chi2 = np.mean(perm_chi2_stats >= chi2_stat)
    perm_p_value_mi = np.mean(perm_mi_scores >= mi)
    
    return {
        'chi2_statistic': chi2_stat,
        'chi2_p_value': p_value,
        'permutation_p_value_chi2': perm_p_value_chi2,
        'mutual_information': mi,
        'permutation_p_value_mi': perm_p_value_mi,
        'contingency_table': orig_contingency,
        'null_mi_distribution': perm_mi_scores,
        'null_chi2_distribution': perm_chi2_stats
    }

def filter_grail_features(mean_alignments, feature_cols, cluster_labels, filter_percentile):
    '''
    Filters features based on the mean alignment of each feature across subjects.
    Filtering happens within clusters, so that features are discarded only if
    they are below threshold in all clusters.
    '''
    # Find the threshold as the nth percentile of absolute values
    threshold = np.percentile(np.abs(mean_alignments), filter_percentile)

    # Create mask for features that have mean(abs) > threshold in at least one cluster
    bool_mask = np.zeros(mean_alignments.shape[1], dtype=bool) # (num_features,)
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        # Get rows for this cluster
        cluster_mask = cluster_labels == label
        cluster_alignments = mean_alignments[cluster_mask]
        
        # Check if mean absolute alignment exceeds threshold for any features
        cluster_means = abs(cluster_alignments).mean(axis=0) # cluster mean for each feature
        bool_mask = bool_mask | (cluster_means > threshold)

    features_filtered = [col for col, mask in zip(feature_cols, bool_mask) if mask]
    return features_filtered

def grail_posthoc_analysis(mean_alignments, num_seeds, seed, filter_percentile=75):
    """
    Perform posthoc analysis on the mean alignments of the Grail experiment.

    Inputs:
    -------
    mean_alignments (pd.DataFrame): Mean alignments of the Grail experiment.
    num_seeds (int): Number of seeds for the clustering.
    seed (int): Random seed for the clustering.
    filter_percentile (int): Percentile to filter features.
    n_permutations (int): Number of permutations for the permutation test.
    """
    feature_cols = list(mean_alignments.columns)
    # 1) Cluster subjects based on their mean feature alignment patterns
    corrmat = np.corrcoef(mean_alignments)
    cluster_labels, _ = louvain_consensus_clustering(corrmat, num_seeds, seed, cmap='YlGnBu')

    # 2) Filter features, keeping only those with mean(abs(alignment)) > n-th percentile
    features_filtered = filter_grail_features(mean_alignments.values, 
                                              feature_cols, 
                                              cluster_labels, 
                                              filter_percentile)
    
    return cluster_labels, features_filtered

# Permutation tests -------------------------------------------------------------

def correlation_permutation_test(y_true, y_pred, n_permutations=1000, seed=None, 
                                 make_plot=True, save_path=None, title=None):
    """
    Perform a permutation test for correlation coefficient.
    
    Parameters:
    -----------
    y_true (np.ndarray): True values (shape: n_samples,)
    y_pred (np.ndarray): Predicted values (shape: n_samples,)
    n_permutations (int): Number of permutations to perform (default: 1000)
    seed (int): Random seed for reproducibility (default: None)
    make_plot (bool): Whether to create a histogram of null distribution (default: True)
    save_path (str): Path to save the plot (default: None)
    title (str): Title for the plot (default: None)
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'observed_r' (float): Observed correlation coefficient
        - 'p_value' (float): Permutation-based p-value
        - 'null_mean' (float): Mean of null distribution
        - 'null_std' (float): Standard deviation of null distribution
        - 'null_distribution' (np.ndarray): Array of permuted correlation values
    """
    # Calculate observed correlation
    observed_r, _ = pearsonr(y_true, y_pred)
    
    # Set up random number generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Perform permutations
    perm_rs = np.empty(n_permutations)
    for i in range(n_permutations):
        # Shuffle labels to simulate random assignment
        y_perm = rng.permutation(y_true)
        try:
            perm_rs[i] = pearsonr(y_perm, y_pred)[0]
        except ValueError:
            # pearsonr can fail if there's insufficient variance
            perm_rs[i] = np.nan
    
    # Drop any failed permutations
    perm_rs = perm_rs[~np.isnan(perm_rs)]
    
    # One-sided p-value: probability |r_null| >= |r_observed|
    # Using absolute value since correlation can be positive or negative
    p_value = (np.sum(np.abs(perm_rs) >= np.abs(observed_r)) + 1) / (len(perm_rs) + 1)
    
    # Calculate null distribution statistics
    null_mean = np.mean(perm_rs)
    null_std = np.std(perm_rs)
    
    # Create plot if requested
    if make_plot:
        plt.figure(figsize=(8, 6))
        sns.histplot(perm_rs, bins=50, kde=True)
        plt.axvline(observed_r, linestyle='--', linewidth=2,
                    color='red', label=f'Observed r = {observed_r:.3f}')
        plt.xlabel('Correlation coefficient under label permutation')
        plt.ylabel('Frequency')
        
        full_title = title or ''
        full_title += (
            f'SD = {null_std:.3f}, '
            f'p = {p_value:.3g}')
        plt.title(full_title)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
    
    return {
        'observed_r': observed_r,
        'p_value': p_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_distribution': perm_rs
    }

def anova_permutation_test(data_groups, n_permutations=10000):
    """
    Perform a permutation test to assess if groups are significantly different.
    
    Parameters:
    -----------
    data_groups (list of arrays): List containing the data for each group
    n_permutations (int): Number of permutations to perform
    
    Returns:
    --------
    f_stat (float): F-statistic for the actual data
    p_value (float): Permutation-based p-value
    """
    # Concatenate all data and get group sizes
    all_data = np.concatenate(data_groups)
    group_sizes = [len(group) for group in data_groups]
    
    # Calculate actual F-statistic
    f_stat, _ = stats.f_oneway(*data_groups)
    
    # Perform permutations
    f_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle the data
        shuffled_data = np.random.permutation(all_data)
        
        # Split into groups of original sizes
        start = 0
        shuffled_groups = []
        for size in group_sizes:
            shuffled_groups.append(shuffled_data[start:start + size])
            start += size
            
        # Calculate F-statistic for permuted data
        f_perm[i], _ = stats.f_oneway(*shuffled_groups)
    
    # Calculate p-value
    p_value = np.mean(f_perm >= f_stat)
    
    return f_stat, p_value

def discrete_variables_permutation_test(var1, var2, n_permutations=10000, save_path=None):
    """
    Test if var1 is predictive of var2 using chi-square test of independence,
    mutual information, and permutation testing.
    
    Parameters:
    -----------
    var1 (numpy.ndarray): First discrete variable (shape: n_samples,)
    var2 (numpy.ndarray): Second discrete variable (shape: n_samples,)
    n_permutations (int): Number of permutations for the test
    
    Returns:
    --------
    dict
        Contains original chi-square statistic, p-values, mutual information,
        and contingency table
    """
    
    # Compute original contingency table and chi-square statistic
    unique_var1 = np.unique(var1)
    unique_var2 = np.unique(var2)
    orig_contingency = np.zeros((len(unique_var1), len(unique_var2)))
    
    for i, val1 in enumerate(unique_var1):
        for j, val2 in enumerate(unique_var2):
            orig_contingency[i, j] = np.sum((var1 == val1) & (var2 == val2))
            
    chi2_stat, p_value, _, _ = stats.chi2_contingency(orig_contingency)
    
    # Compute mutual information
    mi = mutual_info_score(var1, var2)
    
    # Perform permutation test
    perm_chi2_stats = np.zeros(n_permutations)
    perm_mi_scores = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Permute var1
        perm_var1 = np.random.permutation(var1)
        
        # Compute contingency table for permuted data
        perm_contingency = np.zeros((len(unique_var1), len(unique_var2)))
        for j, val1 in enumerate(unique_var1):
            for k, val2 in enumerate(unique_var2):
                perm_contingency[j, k] = np.sum((perm_var1 == val1) & (var2 == val2))
        
        # Compute chi-square statistic for permuted data
        perm_chi2_stats[i], _, _, _ = stats.chi2_contingency(perm_contingency)
        
        # Compute mutual information for permuted data
        perm_mi_scores[i] = mutual_info_score(perm_var1, var2)
    
    # Calculate permutation p-values
    perm_p_value_chi2 = np.mean(perm_chi2_stats >= chi2_stat)
    perm_p_value_mi = np.mean(perm_mi_scores >= mi)

    # Plot the null distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(perm_chi2_stats, bins=50, alpha=0.5)
    plt.axvline(chi2_stat, color='r', linestyle='--', label='Observed')
    plt.title('Null Distribution of Chi-square Statistic')
    plt.xlabel('Chi-square Statistic')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(perm_mi_scores, bins=50, alpha=0.5)
    plt.axvline(mi, color='r', linestyle='--', label='Observed')
    plt.title('Null Distribution of Mutual Information')
    plt.xlabel('Mutual Information')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    results = {'chi2_statistic': chi2_stat,
               'chi2_p_value': p_value,
               'permutation_p_value_chi2': perm_p_value_chi2,
               'mutual_information': mi,
               'permutation_p_value_mi': perm_p_value_mi,
               'contingency_table': orig_contingency,
               'null_mi_distribution': perm_mi_scores,
               'null_chi2_distribution': perm_chi2_stats}

    return results

def test_column_significance(df, test_type='t-test', alpha=0.05):
    """
    Perform statistical tests on each column of a DataFrame to check if values are different from zero.
    
    Args:
        df (pd.DataFrame): DataFrame containing values to test (e.g., z-scores)
        test_type (str): Type of test to perform ('t-test' or 'wilcoxon')
        alpha (float): Significance level for FDR correction
    
    Returns:
        pd.DataFrame: DataFrame containing test statistics for each column
    """
    # Initialize results dictionary
    results = {
        'statistic': [],
        'p_value': [],
        'cohen_d': [],
        'fdr_p_value': [],
        'significant': [],
        'effect_size': []}
    
    # Perform tests for each column
    for col in df.columns:
        values = df[col].values
        
        if test_type == 't-test':
            # One-sample t-test
            t_stat, p_val = stats.ttest_1samp(values, 0)
            results['statistic'].append(t_stat)
            results['p_value'].append(p_val)
            
            # Cohen's d
            cohen_d = np.mean(values) / np.std(values, ddof=1)
            results['cohen_d'].append(cohen_d)
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            w_stat, p_val = stats.wilcoxon(values)
            results['statistic'].append(w_stat)
            results['p_value'].append(p_val)
            
            # For Wilcoxon, we'll use a different effect size measure
            # (r = z/sqrt(N) where z is the standardized test statistic)
            z_stat = stats.norm.ppf(p_val/2)
            n = len(values)
            cohen_d = z_stat / np.sqrt(n)
            results['cohen_d'].append(cohen_d)
    
    # Perform FDR correction
    fdr_p_values = fdrcorrection(np.array(results['p_value']), alpha=alpha)[1]
    results['fdr_p_value'] = fdr_p_values
    
    # Determine significance
    results['significant'] = fdr_p_values < alpha

    # Categorize effect sizes based on Cohen's d
    for cohen_d in results['cohen_d']:
        if abs(cohen_d) < 0.2:
            results['effect_size'].append('small')
        elif abs(cohen_d) < 0.5:
            results['effect_size'].append('medium')
        elif abs(cohen_d) < 0.8:
            results['effect_size'].append('large')
        else:
            results['effect_size'].append('very large')
    
    # Create results DataFrame
    stats_df = pd.DataFrame(results, index=df.columns)
    
    # Rename columns based on test type
    if test_type == 't-test':
        stats_df = stats_df.rename(columns={'statistic': 't_statistic'})
    elif test_type == 'wilcoxon':
        stats_df = stats_df.rename(columns={'statistic': 'w_statistic'})
    
    return stats_df

# Regression model evaluation ---------------------------------------------------

def analyze_coefficient_sparsity(pred_models: Dict, 
                                 n_clinical_features: int) -> pd.DataFrame:
    """
    Calculates the Gini coefficient for the latent feature weights of each t-learner.
    
    This metric quantifies the "non-uniformness" or sparsity of the regression 
    coefficients. A Gini coeff close to 1 indicates the model relies on a few 
    specific latent dimensions (signal). A Gini coeff close to 0 indicates 
    the model spreads weights equally (potential noise fitting).

    Parameters:
    -----------
    pred_models : Dict
        Dictionary {condition: [list of trained sklearn models]} populated during LOOCV.
    n_clinical_features : int
        Number of clinical features concatenated to the end of the input vector.
        These are excluded from the Gini calculation to focus on VGAE latent space.

    Returns:
    --------
    pd.DataFrame containing Gini coefficients and statistics for every fold.
    """
    results = []

    for cond, models in pred_models.items():
        for i, model in enumerate(models):
            # 1. Extract feature weights
            if hasattr(model, 'coef_'):
                # Ridge, ElasticNet, etc.
                weights = np.abs(model.coef_)
                # Handle multi-output case if necessary (flatten), though usually 1D here
                if weights.ndim > 1:
                    weights = np.sum(weights, axis=0) 
            elif hasattr(model, 'feature_importances_'):
                # RandomForest, GradientBoosting
                weights = model.feature_importances_
            else:
                continue # Skip models without accessible weights

            # 2. Separate Latent from Clinical
            # Clinical features are appended at the end
            if n_clinical_features > 0:
                latent_weights = weights[:-n_clinical_features]
            else:
                latent_weights = weights

            # 3. Calculate Gini Coefficient
            # Gini = sum(|xi - xj|) / (2 * n^2 * mean)
            if np.mean(latent_weights) == 0:
                gini = 0.0
            else:
                sorted_w = np.sort(latent_weights)
                n = len(latent_weights)
                index = np.arange(1, n + 1)
                gini = ((2 * np.sum(index * sorted_w)) / (n * np.sum(sorted_w))) - ((n + 1) / n)

            results.append({
                'Condition': cond,
                'Fold_Index': i,
                'Gini_Coefficient': gini,
                'Max_Weight': np.max(latent_weights),
                'Mean_Weight': np.mean(latent_weights),
                'Num_Latent_Dims': len(latent_weights)
            })

    df_results = pd.DataFrame(results)
    return df_results

# PLS-based robustness analysis -------------------------------------------------

def get_fold_performance(results_dir):
    """Calculate performance metrics for each fold from prediction results.
    
    Args:
        results_dir (str): Path to directory containing prediction results
        
    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each fold
    """    
    # Load results
    results = pd.read_csv(os.path.join(results_dir, 'prediction_results.csv'))
    xlearner_test_indices = np.loadtxt(os.path.join(results_dir, 'test_fold_indices.csv'), dtype=int)

    # Sort results by subject_id and add test fold column
    results = results.sort_values('subject_id')
    results['test_fold'] = xlearner_test_indices

    # Calculate metrics for each fold
    fold_performance = {'fold': [], 'r': [], 'p': [], 'rho': [], 'rho_p': [], 'mae': []}
    for fold in results['test_fold'].unique():
        fold_performance['fold'].append(fold)
        fold_mask = results['test_fold'] == fold
        r, p = pearsonr(results.loc[fold_mask, 'prediction'], results.loc[fold_mask, 'label'])
        rho, rho_p = spearmanr(results.loc[fold_mask, 'prediction'], results.loc[fold_mask, 'label'])
        mae = np.mean(np.abs(results.loc[fold_mask, 'prediction'] - results.loc[fold_mask, 'label']))
        fold_performance['r'].append(r)
        fold_performance['p'].append(p)
        fold_performance['rho'].append(rho)
        fold_performance['rho_p'].append(rho_p)
        fold_performance['mae'].append(mae)

    fold_performance = pd.DataFrame(fold_performance)
    fold_performance = fold_performance.sort_values('fold').reset_index(drop=True)
    
    return fold_performance

def get_kfold_features_for_all_subjects(results_paths):
    """
    Returns a list of dataframes of shape (num_folds, num_features) and length 
    num_subjects (== number of rows in each results_path csv file). In other words,
    each dataframe in the output list summarises the alignment patterns of all folds 
    for a single subject.
    
    Args:
        results_paths (list): List of paths to CSV files containing feature values
    
    Returns:
        list: List of DataFrames, where each DataFrame contains feature values across folds for a single subject
    """
    
    # Initialize list to store subject-specific dataframes
    subject_dataframes = []
    
    # Load data from each model
    all_data = []
    for path in results_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data files found in results_paths")
    
    # Get subjects and features from first dataframe
    subjects = all_data[0].index
    features = all_data[0].columns
    
    # For each subject, create a dataframe of their feature values across folds
    for subject in subjects:
        # Create a dataframe for this subject
        subject_df = pd.DataFrame(index=range(len(all_data)), columns=features, dtype=float)
        
        # Fill in the feature values for each fold
        for fold_idx, df in enumerate(all_data):
            subject_df.loc[fold_idx] = df.loc[subject]
        
        subject_dataframes.append(subject_df)
    
    return subject_dataframes

def pls_target_correlation(X_df, y, n_components=1):
    """
    Run PLS to find feature patterns that correlate with a target.

    Parameters:
    ----------
    X_df: pd.DataFrame
        DataFrame of shape (n_samples, n_features)
    y: np.ndarray
        Target vector of shape (n_samples,)
    n_components: int
        Number of PLS components to compute (default: 1)

    Returns:
    --------
    pls: fitted PLSRegression model
    r: Pearson correlation between first component and target
    x_weights: weight vector (feature pattern) of the first PLS component
    """
    X = X_df.values
    y = y.reshape(-1, 1)  # ensure 2D

    # Standardize X and y
    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y)

    # Fit PLS
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_scaled)

    # Get first component scores (projections of samples)
    component_scores = pls.x_scores_[:, 0]
    r, p = pearsonr(component_scores, y_scaled.flatten())
    corr_results = {'r': r, 'p': p}

    pattern_standardised = pls.x_weights_[:, 0]
    pattern_original = pls.x_weights_[:, 0] * X.std(axis=0, ddof=0)
    return pls, pattern_standardised, pattern_original, corr_results

def get_pls_patterns(subject_dataframes, performance):
    """
    Performs PLS to find feature patterns that correlate with performance for each subject.
    Returns a dataframe of shape (num_subjects, num_features) containing the PLS patterns for each subject.
    
    Args:
        subject_dataframes (list): List of DataFrames, where each DataFrame contains feature values across folds for a single subject
        performance (numpy.ndarray): Array of performance values to predict, shape (num_folds,)
    
    Returns:
        pandas.DataFrame: DataFrame of shape (num_subjects, num_features) containing the PLS patterns for each subject
    """
    pls_patterns = []
    corrs_ps = {'sub': [], 'r': [], 'p': []}
    features = subject_dataframes[0].columns

    # For each subject, fit PLS
    for subj_idx, subject_df in enumerate(subject_dataframes):
        # Fit PLS
        _, _, pattern_original, corr_results = pls_target_correlation(subject_df, performance)

        # Store correlation and p-value
        r, p = corr_results['r'], corr_results['p']
        corrs_ps['sub'].append(subj_idx)
        corrs_ps['r'].append(r)
        corrs_ps['p'].append(p)

        # Make sure all patterns are positively correlated with performance
        if r < 0:
            pattern_original = -pattern_original
        
        # Store coefficients
        pls_patterns.append(pattern_original)

    corrs_ps = pd.DataFrame(corrs_ps)
    pls_patterns = pd.DataFrame(pls_patterns, columns=features)
    return pls_patterns, corrs_ps

def load_fold_data(job_dirs, file_names, inclusion_criteria):
    """
    Load and process fold data for multiple jobs.
    
    Parameters
    ----------
    job_dirs : list
        List of full job directory paths
    file_names : list 
        List of CSV file names to load for each fold
    inclusion_criteria : dict
        Dictionary with keys 'metric', 'criterion', 'threshold' specifying inclusion criteria
        
    Returns
    -------
    all_subject_dfs : dict
        Dictionary mapping subject index to concatenated dataframe of features (num_folds x num_features)
    fold_performances : pd.DataFrame
        Concatenated dataframe of fold performances
    """
    # Get number of folds from file_names length
    xlearner_num_folds = len(file_names)
    
    # Initialize output dictionary
    all_subject_dfs = {}
    
    # Load first file to get number of subjects
    first_job = job_dirs[0]
    first_file = os.path.join(first_job, file_names[0])
    if os.path.exists(first_file):
        num_subs = len(pd.read_csv(first_file))
        all_subject_dfs = {sub: [] for sub in range(num_subs)}
    
    fold_performances = []
    missing_jobs = []
    for job_dir in job_dirs:
        # Get subject dataframes (num_folds x num_features)
        results_paths = [os.path.join(job_dir, fname) for fname in file_names]

        # Include only paths that exist
        results_paths = [path for path in results_paths if os.path.exists(path)]
        missing_folds = [k for k in range(xlearner_num_folds) if os.path.join(job_dir, file_names[k]) not in results_paths]
        if len(results_paths) == 0:
            missing_jobs.append(job_dir)
            continue
        subject_dataframes = get_kfold_features_for_all_subjects(results_paths) # list of (num_folds x num_features) dataframes

        # Get fold performance (num_folds, num_performance_metrics)
        fold_performance = pd.read_csv(os.path.join(job_dir, 'fold_performance.csv'))
        fold_performance['job'] = int(job_dir.split('_')[-1])

        # Remove rows in fold_performance that are not in subject_dataframes
        fold_performance = fold_performance[~fold_performance['fold'].isin(missing_folds)]
        
        # Apply inclusion criteria
        if inclusion_criteria['criterion'] == 'greater_than':
            mask = fold_performance[inclusion_criteria['metric']] > inclusion_criteria['threshold']
        elif inclusion_criteria['criterion'] == 'less_than':
            mask = fold_performance[inclusion_criteria['metric']] < inclusion_criteria['threshold']
        fold_performance = fold_performance[mask]
        
        # Keep only corresponding rows in subject_dataframes
        included_folds = fold_performance['fold'].values
        subject_dataframes = [df[df.index.isin(included_folds)] for df in subject_dataframes]

        # Store subject dataframes and fold performances
        for subj_idx, subject_df in enumerate(subject_dataframes):
            if not subject_df.empty:  # Only store if there are rows remaining
                all_subject_dfs[subj_idx].append(subject_df)
        fold_performances.append(fold_performance)

    # Concatenate all subject dataframes and fold performances
    for sub in range(num_subs):
        if sub in all_subject_dfs and len(all_subject_dfs[sub]) > 0:
            all_subject_dfs[sub] = pd.concat(all_subject_dfs[sub], axis=0)
            
    fold_performances = pd.concat(fold_performances, axis=0)

    # Print the number of CV-fold models that were included in the analysis
    max_num_models = xlearner_num_folds * len(job_dirs)
    print(f'CV-fold models included: {len(fold_performances)} out of {max_num_models}')
            
    return all_subject_dfs, fold_performances  

def pls_robustness_analysis(all_subject_dfs: dict[int, pd.DataFrame], 
                            performance: np.ndarray):
    """
    Performs PLS to find feature GRAIL patterns that correlate with test performance.

    Parameters
    ----------
    all_subject_dfs (dict[int, pd.DataFrame]): Dictionary mapping subject index to 
        concatenated dataframe of features (num_folds x num_features)
    performance (np.ndarray): Array of performance values to predict, shape (num_folds,)

    Returns
    -------
    pls_patterns (pd.DataFrame): DataFrame of shape (num_subjects, num_features) containing 
        the PLS patterns for each subject.
    pls_performance_corrs (pd.DataFrame): DataFrame of shape (num_subjects, 3) containing 
        the correlation and p-value of each subject's PLS pattern with performance.
    pls_weight_stats (pd.DataFrame): DataFrame of shape (num_features, 8) containing 
        the t-statistic, p-value, cohen's d, fdr_p_value, significant, and effect_size 
        for each feature (i.e. candidate biomarker or brain region).
    """
    # Get PLS patterns for each subject
    # (= patterns that maximally correlate with fold-model performance)
    subject_df_list = [df for df in all_subject_dfs.values()]
    pls_patterns, pls_performance_corrs = get_pls_patterns(subject_df_list, performance)

    # Test if features are reliably associated with performance across subjects
    # (i.e. if the PLS patterns are significantly different from zero)
    pls_weight_stats = test_column_significance(pls_patterns, test_type='t-test')

    return pls_patterns, pls_performance_corrs, pls_weight_stats

def compute_performance_weighted_means(all_subject_dfs: dict[int, pd.DataFrame],
                                       performance: np.ndarray,
                                       performance_cutoff: float = 0.0):
    """
    Computes performance-weighted means for all features across all subjects.

    Parameters
    ----------
    all_subject_dfs (dict[int, pd.DataFrame]): Dictionary mapping subject index to 
        concatenated dataframe of features (num_folds x num_features)
    performance (np.ndarray): Array of performance values to predict, shape (num_folds,)
    performance_cutoff (float): Minimum performance value to consider for weighting

    Returns
    -------
    weighted_means (pd.DataFrame): DataFrame of shape (num_subjects, num_features) 
        containing the performance-weighted means for each feature.
    weighted_means_stats (pd.DataFrame): DataFrame of shape (num_features, 8) containing 
        the t-statistic, p-value, cohen's d, fdr_p_value, significant, and effect_size 
        for each feature.
    """
    # Compute preformance-weighted mean alignments for all subjects and features
    weighted_means = []
    feature_names = all_subject_dfs[0].columns
    subject_ids = []
    for sub_id, subject_alignments in all_subject_dfs.items():
        subject_ids.append(sub_id)
        performance[performance < performance_cutoff] = 0
        weights = performance / (np.sum(performance) + 1e-6) # shape (num_folds, )
        weighted_alignments = subject_alignments * weights[:, np.newaxis] # shape (num_folds, num_features)
        wm = np.sum(weighted_alignments, axis=0) # shape (num_features, )
        weighted_means.append(wm)

    # Turn into dataframe with features as columns
    weighted_means = pd.DataFrame(weighted_means, 
                                  index=subject_ids, 
                                  columns=feature_names, dtype=float)
    
    # Test if features have significantly non-zero means in well-performing models
    weighted_means_stats = test_column_significance(weighted_means, test_type='t-test')

    return weighted_means, weighted_means_stats

def pls_feature_filtering(pls_weight_stats: pd.DataFrame, 
                          weighted_means_stats: pd.DataFrame,
                          filter_criteria: dict):
    """
    Filters features based on PLS weight stats and weight-mean alignments.
    Especially relevant is features are signed to guarantee interpretability.

    Parameters
    ----------
    pls_weight_stats (pd.DataFrame): DataFrame of shape (num_features, 8) with columns 
        t_statistic, p_value, cohen_d, fdr_p_value, significant, and effect_size and row
        indices corresponding to features.
    weighted_means_stats (pd.DataFrame): DataFrame of shape (num_features, 8) with columns 
        t_statistic, p_value, cohen_d, fdr_p_value, significant, and effect_size and row
        indices corresponding to features.
    filter_criteria (dict): Dictionary with keys 'pls_weight_stats', 'weighted_means_stats', 
        and 'signed_features' specifying the criteria for filtering features.

    Returns
    -------
    feature_list (list): List of features that meet the criteria.
    """
    # Get features that meet PLS weight criteria
    pls_criteria = filter_criteria['pls_weight_stats']
    pls_features = pls_weight_stats[
        (pls_weight_stats['fdr_p_value'] <= pls_criteria['fdr_p_value'])
    ]
    
    if 'cohen_d' in pls_criteria:
        pls_features = pls_features[
            abs(pls_features['cohen_d']) >= pls_criteria['cohen_d']
        ]
    
    # Get features that meet weighted means criteria
    means_criteria = filter_criteria['weighted_means_stats']
    means_features = weighted_means_stats[
        (weighted_means_stats['fdr_p_value'] <= means_criteria['fdr_p_value'])
    ]
    
    if 'cohen_d' in means_criteria:
        means_features = means_features[
            abs(means_features['cohen_d']) >= means_criteria['cohen_d']
        ]
    
    # Get intersection of features that meet both criteria
    common_features = set(pls_features.index) & set(means_features.index)
    
    # If signed features are required, check sign alignment
    if filter_criteria.get('signed_features', False):
        signed_features = []
        for feature in common_features:
            pls_sign = np.sign(pls_weight_stats.loc[feature, 't_statistic'])
            means_sign = np.sign(weighted_means_stats.loc[feature, 't_statistic'])
            if pls_sign == means_sign:
                signed_features.append(feature)
        return signed_features
    
    return list(common_features)

def calculate_subject_consistency(
    results_base_dir: str, 
    filenames: List[str], 
    subdir_name_pattern: str = 'seed_*',
    relevant_columns: List[str] = None
) -> pd.DataFrame:
    """
    Loads results from multiple subdirectories and files, treating each specific file 
    in each specific subdirectory as an independent model run.

    It calculates the row-wise (subject-wise) correlation consistency across ALL 
    loaded model runs (seeds * filenames).

    For each subject (row), it calculates the correlation of their data vector between 
    all unique pairs of loaded runs.

    Args:
        results_base_dir: Path to the parent directory containing result subfolders.
        filenames: List of filenames to load within each subdirectory. 
                   Each file is treated as an independent model.
        subdir_name_pattern: Glob pattern to identify subdirectories (default: 'seed_*').
        relevant_columns: Optional[List[str]]; if provided, only consider these columns for correlation.

    Returns:
        pd.DataFrame: DataFrame with index as subject ID and columns:
                      ['mean_corr', 'std_corr', 'se_corr'].
    """
    
    # 1. Find Subdirectories
    subdirs = sorted([
        d for d in glob.glob(os.path.join(results_base_dir, subdir_name_pattern))
        if os.path.isdir(d)
    ])

    if not subdirs:
        raise ValueError(f"No subdirectories found matching {subdir_name_pattern}")

    print(f"Found {len(subdirs)} directories. Scanning for {len(filenames)} files per directory...")

    all_run_dfs = []
    reference_columns = None
    
    # 2. Load Data: Flatten structure (Seeds x Folds)
    for d in subdirs:
        for fname in filenames:
            fpath = os.path.join(d, fname)
            
            if not os.path.exists(fpath):
                print(f"Warning: Missing file {fname} in {d}. Skipping this specific run.")
                continue
            
            try:
                df = pd.read_csv(fpath)
                
                # Optionally select relevant_columns
                if relevant_columns is not None:
                    missing_cols = set(relevant_columns) - set(df.columns)
                    if missing_cols:
                        print(f"Warning: Missing columns {missing_cols} in {fpath}. Skipping this run.")
                        continue
                    df = df[relevant_columns]
                
                if reference_columns is None:
                    reference_columns = df.columns.tolist()
                else:
                    # Reorder columns to match reference to ensure feature alignment
                    # If columns are missing or different, this helps catch it or align it
                    if set(df.columns) != set(reference_columns):
                        print(f"Warning: Column mismatch in {fpath}. Aligning to reference.")
                    # Select columns in correct order
                    df = df[reference_columns]

                all_run_dfs.append(df)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    num_total_runs = len(all_run_dfs)
    if num_total_runs < 2:
        raise ValueError(f"Insufficient valid data loaded. Needed >= 2 runs, found {num_total_runs}.")

    # Check shapes
    base_shape = all_run_dfs[0].shape
    for i, df in enumerate(all_run_dfs):
        if df.shape != base_shape:
            raise ValueError(f"Shape mismatch in loaded run index {i}. Expected {base_shape}, got {df.shape}")

    print(f"Computing consistency across {num_total_runs} total independent models (Seeds x Folds) for {base_shape[0]} subjects...")

    # 3. Compute Correlations
    # We convert to a 3D numpy array: (Num_Runs, Num_Subjects, Num_Features)
    data_stack = np.stack([df.values for df in all_run_dfs])
    
    n_subjects = base_shape[0]
    results = []
    
    # Generate all unique pairs of run indices (e.g., (0,1), (0,2)... (n, n-1))
    run_pairs = list(itertools.combinations(range(num_total_runs), 2))
    
    for subject_idx in range(n_subjects):
        correlations = []
        
        # Extract the vector for this subject across all runs
        # Shape: (Num_Runs, Num_Features)
        subject_vectors = data_stack[:, subject_idx, :]
        
        # Calculate correlation for every pair of runs
        for run_i, run_j in run_pairs:
            vec_a = subject_vectors[run_i]
            vec_b = subject_vectors[run_j]
            
            # Handle constant vectors (std=0) which cause NaN correlation
            if np.std(vec_a) == 0 or np.std(vec_b) == 0:
                corr = np.nan
            else:
                # np.corrcoef returns [[1, r], [r, 1]]
                corr = np.corrcoef(vec_a, vec_b)[0, 1]
                
            correlations.append(corr)
            
        # Compute Stats
        correlations = np.array(correlations)
        
        mean_r = np.nanmean(correlations)
        std_r = np.nanstd(correlations)
        
        count_valid = np.sum(~np.isnan(correlations))
        se_r = std_r / np.sqrt(count_valid) if count_valid > 0 else np.nan
        
        results.append({
            'subject': subject_idx,
            'mean_corr': mean_r,
            'std_corr': std_r,
            'se_corr': se_r
        })

    return pd.DataFrame(results)

def elasticnet_cv_predict(X, y, subject_ids=None, n_splits=7, random_state=0, verbose=True):
    """
    Run k-fold CV ElasticNet prediction given X, y, [optional] conditions vector.
    Returns a DataFrame with columns: 'prediction', 'label', and 'Condition' (if provided).
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_preds_all = []
    y_true_all = []
    subject_ids_all = []

    # Loop over folds
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = ElasticNetCV(cv=5, random_state=random_state, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        y_preds_all.extend(preds)
        y_true_all.extend(y_test)
        if subject_ids is not None:
            subject_ids_all.extend(subject_ids[test_idx])
        else:
            subject_ids_all.extend([None]*len(test_idx))
        if verbose:
            print(f"Fold {fold_i+1}/{n_splits} complete.")

    y_true_all = np.array(y_true_all)
    y_preds_all = np.array(y_preds_all)
    subject_ids_all = np.array(subject_ids_all)

    r_value, _ = pearsonr(y_true_all, y_preds_all)
    r2_val = r2_score(y_true_all, y_preds_all)

    if verbose:
        print("\n--- Final Results (Aggregated) ---")
        print(f"Pearson Correlation (r): {r_value:.3f}")
        print(f"Coefficient of Determination (R²): {r2_val:.3f}")

    df = pd.DataFrame({'prediction': y_preds_all, 'label': y_true_all, 'subject_id': subject_ids_all})
    df = df.sort_values('subject_id').reset_index(drop=True)
    return df