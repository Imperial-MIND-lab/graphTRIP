"""
Util functions for statistical analysis and algorithms.

Author: Hanna M. Tolle
Date: 2025-02-12
License: BSD-3-Clause
"""

import sys
import os
sys.path.append('graphTRP/')

from dominance_analysis import Dominance
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency


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

def grail_posthoc_analysis(mean_alignments, num_seeds, seed, filter_percentile=75, n_permutations=10000):
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

# Surrogate alignment/ attention tests ------------------------------------------

def compute_feature_z_scores(surrogate_dir, observed_dir, results_file, epsilon=1e-6):
    """
    Compute z-scores for each feature and subject by comparing each subject's observed value
    against the distribution of surrogate values for that specific subject.
    
    Args:
        surrogate_dir (str): Directory containing subdirectories with surrogate results
        observed_dir (str): Directory containing observed results
        results_file (str): Name of the results file in each directory
        epsilon (float): Small value to prevent division by zero in std calculation
    
    Returns:
        pd.DataFrame: DataFrame containing z-scores for each feature and subject
    """
    # Read observed data
    observed_path = os.path.join(observed_dir, results_file)
    observed_df = pd.read_csv(observed_path)
    
    # Drop 'Condition' column if it exists
    if 'Condition' in observed_df.columns:
        observed_df = observed_df.drop('Condition', axis=1)
    
    # Initialize dictionary to store surrogate values for each subject and feature
    # surrogate_values[subject_idx][feature] = list of surrogate values
    surrogate_values = {idx: {col: [] for col in observed_df.columns} 
                       for idx in observed_df.index}
    
    # Collect surrogate values for each subject from all job directories
    for job_dir in os.listdir(surrogate_dir):
        if job_dir.startswith('job_'):
            job_path = os.path.join(surrogate_dir, job_dir, results_file)
            if os.path.exists(job_path):
                surrogate_df = pd.read_csv(job_path)
                if 'Condition' in surrogate_df.columns:
                    surrogate_df = surrogate_df.drop('Condition', axis=1)
                
                # For each subject, collect their surrogate values
                for idx in observed_df.index:
                    for col in observed_df.columns:
                        surrogate_values[idx][col].append(surrogate_df.loc[idx, col])
    
    # Initialize z-score array
    z_scores_array = np.zeros((len(observed_df.index), len(observed_df.columns)))
    
    # Compute z-scores for each subject and feature
    for i, idx in enumerate(observed_df.index):
        for j, col in enumerate(observed_df.columns):
            # Get observed value for this subject and feature
            observed_value = observed_df.loc[idx, col]
            
            # Get surrogate values for this subject and feature
            subject_surrogate_values = np.array(surrogate_values[idx][col])
            
            # Compute mean and std of surrogate values for this subject
            surrogate_mean = np.mean(subject_surrogate_values)
            surrogate_std = np.std(subject_surrogate_values)
            
            # Compute z-score
            z_scores_array[i, j] = (observed_value - surrogate_mean) / (surrogate_std + epsilon)
    
    # Create DataFrame with proper numeric dtype
    z_scores = pd.DataFrame(z_scores_array, 
                          index=observed_df.index, 
                          columns=observed_df.columns,
                          dtype=float)
    
    return z_scores

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
        else:
            results['effect_size'].append('large')
    
    # Create results DataFrame
    stats_df = pd.DataFrame(results, index=df.columns)
    
    # Rename columns based on test type
    if test_type == 't-test':
        stats_df = stats_df.rename(columns={'statistic': 't_statistic'})
    elif test_type == 'wilcoxon':
        stats_df = stats_df.rename(columns={'statistic': 'w_statistic'})
    
    return stats_df

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
        Dictionary mapping subject index to concatenated dataframe of features
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
            
    return all_subject_dfs, fold_performances