"""
Analysis and plotting routines shared by several figure targets.

Each function here replaces a block that was copy-pasted between notebook cells; the
docstrings name the cells they came from.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, t as t_dist
from statsmodels.stats.multitest import fdrcorrection

from utils.helpers import aggregate_prediction_results, sort_features
from utils.statsalg import min_significant_r, compare_model_performances
from utils.plotting import (
    COOLWARM, ESCIT, PSILO,
    true_vs_pred_scatter, plot_raincloud, plot_metric_boxplot,
    plot_fc_reconstruction_single, plot_brain_surface_grid, plot_colormap_stack,
    plot_stacked_percentages, plot_piechart, plot_confusion_matrix, plot_roc_curve)

from figure_making.paths import require, biomarker_categories_file


FC_VRANGE = (-0.7, 0.7)
EXAMPLE_SUBJECT = 0


# Prediction scatters ------------------------------------------------------------------

def scatter_from_results(results_file, out, name, condition_study=None, **kwargs):
    '''
    Aggregates prediction results across seeds and plots true versus predicted.

    Replaces the aggregate_prediction_results + true_vs_pred_scatter pair, which the
    notebook repeats 26 times.

    Parameters:
    ----------
        results_file (str): Aggregated results CSV; seed_*/ subdirs are aggregated
                            into it if it does not exist yet.
        out (FigureOutput): Output handle of the calling target.
        name (str): Panel filename without extension.
        condition_study (str): If given, adds the drug condition column for this study.

    Returns:
    -------
        pd.DataFrame: The aggregated results.
    '''
    require(os.path.dirname(results_file))
    results = aggregate_prediction_results(results_file=results_file)

    if condition_study is not None:
        from experiments.ingredients.data_ingredient import add_drug_condition_to_outputs
        results = add_drug_condition_to_outputs(results, condition_study)

    true_vs_pred_scatter(results, save_path=out.fig(name), **kwargs)
    return results


# Partial correlations (notebook cells 22, 24, 25) ---------------------------------------

PARTIAL_CORR_ANALYSES = [
    {'name': 'Condition only',
     'covariates': ['Condition'],
     'suffix': 'condition'},
    {'name': 'QIDS_Before only',
     'covariates': ['QIDS_Before'],
     'suffix': 'qids_before'},
    {'name': 'Condition + QIDS_Before',
     'covariates': ['Condition', 'QIDS_Before'],
     'suffix': 'condition_qids_before'},
]


def _residualize(df, y_col, covariate_cols):
    X = sm.add_constant(df[list(covariate_cols)], has_constant="add")
    return sm.OLS(df[y_col], X).fit().resid


def partial_correlation(df, x_col, y_col, covariate_cols):
    '''Returns (r, p, y_residuals, x_residuals) after regressing out the covariates.'''
    x_res = _residualize(df, x_col, covariate_cols)
    y_res = _residualize(df, y_col, covariate_cols)

    r, _ = pearsonr(x_res, y_res)
    n = len(df)
    k = len(covariate_cols)
    df_resid = n - k - 2
    t_stat = r * np.sqrt(df_resid / max(1.0 - r**2, 1e-12))
    p = 2 * t_dist.sf(np.abs(t_stat), df=df_resid)

    return r, p, y_res, x_res


def partial_correlation_panels(results, out, prefix, xcol='label', ycol='prediction',
                               yerr=None):
    '''
    Runs the three partial-correlation analyses of prediction versus label and plots each.

    Controls for treatment condition, baseline QIDS, and both.
    '''
    summary_rows = []

    for analysis in PARTIAL_CORR_ANALYSES:
        covs = analysis['covariates']
        r, p, y_res, x_res = partial_correlation(results, ycol, xcol, covs)

        out.log(f"Partial correlation ({analysis['name']}):")
        out.log(f"   covariates: {', '.join(covs)}")
        out.log(f"   r = {r:.4f}")
        out.log(f"   p = {p:.4e}")
        out.log()

        x_resid = f"{xcol}_controlled_{analysis['suffix']}"
        y_resid = f"{ycol}_controlled_{analysis['suffix']}"

        plot_df = results.copy()
        plot_df[x_resid] = y_res
        plot_df[y_resid] = x_res

        true_vs_pred_scatter(plot_df, xcol=x_resid, ycol=y_resid, yerr=yerr,
                             save_path=out.fig(f"{prefix}_{analysis['suffix']}"))

        summary_rows.append({'analysis': analysis['name'],
                             'covariates': ', '.join(covs),
                             'r': r,
                             'p': p})

    return pd.DataFrame(summary_rows)


# Reconstructions (notebook cells 29-31, 41-43, 111-113) ---------------------------------

def plot_reconstruction_panels(ctx, out, recon, atlas, rsn_mapping, rsn_labels, conditions,
                               brain_subdir, suffix='', sub=EXAMPLE_SUBJECT,
                               fc_name=None, corr_name=None):
    '''
    Plots the three reconstruction panels for one dataset/atlas and saves the regional
    values needed for surface rendering.

    Panels: the FC reconstruction matrix, the node-attribute surface grid, and the
    original-versus-reconstructed correlation boxplot.

    Parameters:
    ----------
        recon (tuple): (adj_orig_rcn, x_orig_rcn) from ctx.reconstructions().
        brain_subdir (str): Subdirectory of brain_plots/ for the regional value CSVs.
        suffix (str): Appended to the FC and correlation panel names, e.g. '_schaefer200'.
        fc_name, corr_name: Override the default panel names where the notebook used
                            a different one.
    '''
    adj_orig_rcn, x_orig_rcn = recon

    # FC reconstruction matrix
    fc_name = fc_name or f'fc_reconstruction_sub{sub}{suffix}'
    plot_fc_reconstruction_single(adj_orig_rcn,
                                  rsn_mapping=rsn_mapping,
                                  rsn_labels=rsn_labels,
                                  subject_idx=sub,
                                  cmap=COOLWARM,
                                  vrange=FC_VRANGE,
                                  save_path=out.fig(fc_name))

    # Node attributes on the brain surface. Always raster, regardless of --fmt.
    reconstructed = x_orig_rcn['reconstructed'][:, :, sub]
    original = x_orig_rcn['original'][:, :, sub]
    data2plot = np.stack([reconstructed, original], axis=0)
    plot_brain_surface_grid(data2plot,
                            atlas=atlas,
                            view='medial',
                            cmap=ctx.x_cmaps,
                            column_names=x_orig_rcn['feature_names'],
                            save_path=out.fig(f'x_reconstructions_sub{sub}', ext='png'))

    # Regional values for surface rendering with MATLAB BrainNetViewer
    feature_names = x_orig_rcn['feature_names']
    out.brain(f'{brain_subdir}/reconstructed_sub{sub}',
              pd.DataFrame(reconstructed, columns=feature_names))
    out.brain(f'{brain_subdir}/original_sub{sub}',
              pd.DataFrame(original, columns=feature_names))

    # Node attribute and FC correlations together
    corr_name = corr_name or f'original_vs_reconstructed_corrs{suffix}'
    plot_metric_boxplot(x_orig_rcn['metrics']['corr'],
                        conditions=conditions,
                        short_names=True,
                        ylabel='correlation',
                        save_path=out.fig(corr_name))


def plot_correlation_boxplot(out, x_orig_rcn, conditions, name, short_names=True):
    '''Plots the original-versus-reconstructed correlations for node features and FC.'''
    plot_metric_boxplot(x_orig_rcn['metrics']['corr'],
                        conditions=conditions,
                        short_names=short_names,
                        ylabel='correlation',
                        save_path=out.fig(name))


# Seed metrics and model comparison (cells 54-56, 82-84, 89-91, 94-96, 120) ---------------

def collect_seed_metrics(specs, skip_missing=False):
    '''
    Collects per-seed performance metrics for several models into one dataframe.

    Parameters:
    ----------
        specs (list): (label, directory, metrics_filename) triples. Each directory is
                      expected to contain seed_*/ subdirectories.
        skip_missing (bool): Skip models whose directory does not exist, instead of
                             raising MissingInput.

    Returns:
    -------
        pd.DataFrame: Concatenated metrics with an added 'model' column.
    '''
    dfs = []
    for label, subdir_path, metrics_filename in specs:
        if skip_missing and not os.path.exists(subdir_path):
            continue
        require(subdir_path)
        seed_dirs = sorted(d for d in os.listdir(subdir_path)
                           if d.startswith('seed_') and os.path.isdir(os.path.join(subdir_path, d)))
        for seed_dir in seed_dirs:
            metrics_path = os.path.join(subdir_path, seed_dir, metrics_filename)
            if not os.path.exists(metrics_path):
                print(f'Warning: Metrics file {metrics_path} does not exist')
                continue
            df = pd.read_csv(metrics_path)
            df['model'] = label
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f'No metrics files found for {[s[0] for s in specs]}')
    return pd.concat(dfs, ignore_index=True)


def metrics_to_distributions(metrics_df, sort_by_mean=False, metric='r'):
    '''Converts the output of collect_seed_metrics into {model: [values]} for plotting.'''
    distributions = {model: metrics_df[metrics_df['model'] == model][metric].tolist()
                     for model in metrics_df['model'].unique()}
    if sort_by_mean:
        distributions = dict(sorted(distributions.items(), key=lambda x: np.mean(x[1])))
    return distributions


def raincloud_of_model_r(distributions, out, name, num_subs, offset=4, figsize=(8, 4)):
    '''
    Plots the distribution of correlation coefficients across seeds for each model,
    with the significance threshold marked.
    '''
    r_min = min_significant_r(num_subs)
    out.log(f'Minimum significant r-value: {r_min}')

    colors = plot_colormap_stack('YlGnBu', len(distributions) + offset, make_plot=False)
    colors = colors[offset:]
    palette = {name_: color for name_, color in zip(distributions.keys(), colors)}

    plot_raincloud(distributions,
                   palette=palette,
                   save_path=out.fig(name),
                   alpha=0.5,
                   box_alpha=0.3,
                   vline=r_min,
                   sort_by_mean=False,
                   figsize=figsize)


def report_model_comparison(distributions, out, model_of_interest, table_prefix=''):
    '''
    Tests performance differences between the model of interest and the others, and
    writes the global effect and pairwise comparisons to CSV.
    '''
    global_df, posthoc_df = compare_model_performances(
        distributions, is_dependent=True, model_of_interest=model_of_interest)

    out.log_df('Global Effect', global_df)

    of_interest = ((posthoc_df['Model A'] == model_of_interest) |
                   (posthoc_df['Model B'] == model_of_interest))
    out.log_df('Pairwise Comparisons', posthoc_df[of_interest])

    for model, values in distributions.items():
        values = pd.Series(values)
        out.log(f'{model}: mean r = {values.mean():.3f}, sem r = {values.sem():.3f}')

    out.table(f'{table_prefix}global_effect', global_df)
    out.table(f'{table_prefix}pairwise_comparisons', posthoc_df)
    return global_df, posthoc_df


def model_comparison_panels(specs, out, name, num_subs, model_of_interest,
                            sort_by_mean=True, offset=4, figsize=(8, 4),
                            skip_missing=False, table_prefix=''):
    '''Runs the full seed-sensitivity block: collect metrics, raincloud, comparison stats.'''
    metrics_df = collect_seed_metrics(specs, skip_missing=skip_missing)
    distributions = metrics_to_distributions(metrics_df, sort_by_mean=sort_by_mean)
    raincloud_of_model_r(distributions, out, name, num_subs, offset=offset, figsize=figsize)
    report_model_comparison(distributions, out, model_of_interest, table_prefix=table_prefix)
    return distributions


# Biomarker categories (cells 74, 143, 145) ----------------------------------------------

BIOMARKER_CATEGORIES = [
    'Shared_response',
    'Shared_resistance',
    'E_response',
    'P_response',
    'E_resistance',
    'P_resistance',
    'E_response_P_resistance',
    'P_response_E_resistance',
]


def biomarker_palette():
    '''Returns the colour palette for the biomarker categories.'''
    colors = sns.color_palette("Spectral_r", len(BIOMARKER_CATEGORIES))
    palette = {name: color for name, color in zip(BIOMARKER_CATEGORIES, colors)}
    palette['n.s.'] = (0.9, 0.9, 0.9)
    return palette


def load_biomarker_categories(thresh=0.5):
    '''
    Loads the per-subject biomarker categorisations, filters and orders them.

    Biomarkers are kept when at most thresh (as a fraction of subjects) are 'n.s.';
    thresh=1.0 therefore keeps everything except all-'n.s.' biomarkers.

    The ordering is centralised here: categories are sorted in reverse alphabetical
    order and features within a category by sort_features(). The notebook used an
    unordered set() for the supplementary heatmap, which was not reproducible across
    runs; that panel's column order therefore changes, while Fig.6f is unaffected.

    Returns:
    -------
        tuple: (categories_df, majority_cat, sorted_biomarkers)
    '''
    categories = pd.read_csv(require(biomarker_categories_file()))

    num_ns = categories.apply(lambda col: col.value_counts().get('n.s.', 0))
    if thresh >= 1.0:
        categories = categories.loc[:, num_ns < len(categories)]
    else:
        categories = categories.loc[:, num_ns <= int(thresh * len(categories))]

    # Majority (non-'n.s.') category of each biomarker
    majority_cat = {}
    for biomarker in categories.columns:
        counts = categories[biomarker].value_counts()
        if 'n.s.' in counts:
            counts = counts.drop('n.s.')
        majority_cat[biomarker] = counts.idxmax() if not counts.empty else 'n.s.'

    # Sort biomarkers by category, then within category
    unique_cats = sorted(set(majority_cat.values()), reverse=True)
    cat_biomarker_mapping = {cat: [] for cat in unique_cats}
    for feat, cat in majority_cat.items():
        cat_biomarker_mapping[cat].append(feat)

    sorted_biomarkers = []
    for cat in unique_cats:
        sorted_biomarkers.extend(sort_features(cat_biomarker_mapping[cat]))

    return categories, majority_cat, sorted_biomarkers


# Regional attributions in resting-state networks (cells 66, 67) --------------------------

def rsn_attribution_panels(results_dir, out, name, offset=3, figsize=(5, 5)):
    '''
    Plots the distribution of regional attributions per resting-state network and tests
    every pair of networks against each other.
    '''
    rsn_df = pd.read_csv(os.path.join(require(results_dir), 'weighted_mean_rsn_attributions.csv'))
    rsn_names = list(rsn_df.columns)
    rsn_dict = rsn_df.to_dict(orient='list')

    colors = sns.color_palette("YlGnBu_r", len(rsn_names) + offset)
    palette = {name_: color for name_, color in zip(rsn_dict.keys(), colors)}
    plot_raincloud(rsn_dict,
                   palette=palette,
                   save_path=out.fig(name),
                   alpha=0.7,
                   box_alpha=0.5,
                   figsize=figsize,
                   sort_by_mean=False)

    # Compare each pair of RSNs with a paired t-test
    rows = {'RSN1': [], 'RSN2': [], 'tstat': [], 'pval': [], 'cohen_d': []}
    for i, rsn1 in enumerate(rsn_names):
        for rsn2 in rsn_names[i + 1:]:
            t_stat, p_val = stats.ttest_rel(rsn_dict[rsn1], rsn_dict[rsn2])
            d = (np.mean(rsn_dict[rsn1]) - np.mean(rsn_dict[rsn2])) / \
                np.sqrt((np.std(rsn_dict[rsn1])**2 + np.std(rsn_dict[rsn2])**2) / 2)
            rows['RSN1'].append(rsn1)
            rows['RSN2'].append(rsn2)
            rows['tstat'].append(t_stat)
            rows['pval'].append(p_val)
            rows['cohen_d'].append(d)

    stats_df = pd.DataFrame(rows)
    stats_df['fdr'] = fdrcorrection(stats_df['pval'])[1]
    stats_df = stats_df.sort_values('pval')
    return stats_df


# Dominance analysis (cells 69-70, 71-72) ------------------------------------------------

def dominance_panels(results_dir, out, prefix, offset=5, figsize=(8, 2)):
    '''
    Plots the relative importance of receptor maps and the unimodal-transmodal axis in
    explaining regional attributions, as stacked percentages and as a pie chart.
    '''
    results_dir = require(results_dir)
    da_df = pd.read_csv(os.path.join(results_dir, 'da_receptors_utaxis_stats.csv'), index_col=0)

    palette = sns.color_palette("YlGnBu", len(da_df) + offset)[:-offset]
    plot_stacked_percentages(df=da_df,
                             percentage_col='Percentage Relative Importance',
                             save_path=out.fig(f'{prefix}_dominance_analysis'),
                             palette=palette,
                             figsize=figsize)

    plot_piechart(df=da_df,
                  percentage_col='Percentage Relative Importance',
                  save_path=out.fig(f'{prefix}_dominance_analysis_piechart'),
                  palette=sns.color_palette("YlGnBu_r", len(da_df)),
                  figsize=figsize,
                  alpha=0.9)

    metrics = pd.read_csv(os.path.join(results_dir, 'da_receptors_utaxis_r2_pval_tstat.csv'))
    out.log(f"{prefix} dominance analysis: "
            f"R2 = {metrics['r2'].values[0]}, p-value = {metrics['p_value'].values[0]}")
    return da_df


# Propensity estimation (cells 129, 130) -------------------------------------------------

def propensity_panels(results_dir, out, suffix=''):
    '''
    Plots how well the drug condition can be predicted from the baseline data: confusion
    matrix, ROC curve, and the distribution of propensities per condition.
    '''
    results_file = os.path.join(require(results_dir), 'prediction_results.csv')
    results = aggregate_prediction_results(results_file)

    plot_confusion_matrix(results['prediction'], results['label'], threshold=0.5,
                          save_path=out.fig(f'confusion_matrix{suffix}'))
    plot_roc_curve(results['prediction'], results['label'],
                   save_path=out.fig(f'auc_curve{suffix}'))

    distributions = {'escitalopram': results[results['label'] == 0]['prediction'],
                     'psilocybin': results[results['label'] == 1]['prediction']}
    plot_raincloud(distributions,
                   palette={'escitalopram': ESCIT, 'psilocybin': PSILO},
                   vline=0.5,
                   box_alpha=0.5,
                   figsize=(4, 5),
                   save_path=out.fig(f'propensity_raincloud{suffix}'))
    return results
