"""
Analysis and plotting routines shared by several figure targets.

Each function here replaces a block that was copy-pasted between notebook cells; the
docstrings name the cells they came from.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, norm, t as t_dist
from statsmodels.stats.multitest import fdrcorrection

from utils.helpers import (
    aggregate_importance_scores, aggregate_prediction_results, sort_features,
    summarise_seed_metrics)
from utils.statsalg import (
    min_significant_r, compare_model_performances, compute_within_group_pearsonr)
from utils.plotting import (
    ALPHA_SCATTER, BOX_COLOR, COOLWARM, ESCIT, PSILO, NEUTRAL,
    true_vs_pred_scatter, plot_raincloud, plot_metric_boxplot,
    permutation_importance_bar_chart,
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


# Prediction accuracy as tables ----------------------------------------------------------

# Condition is coded identically in both studies: 1 = psilocybin, -1 = escitalopram.
# Psilocybin first, so that the Fisher z below contrasts escitalopram against psilocybin.
ARM_NAMES = {1.0: 'Psilocybin', -1.0: 'Escitalopram'}


def fmt_p(pval):
    '''
    Formats a p-value for a statistics report: a 4-decimal float down to 0.0001, then
    scientific with one decimal.
    '''
    return f'{pval:.4f}' if pval >= 1e-4 else f'{pval:.1e}'


def fmt_p_floor(pval, decimals=3):
    '''
    Formats a p-value for a panel title, without rounding small values to zero. Unlike
    fmt_p, values below the floor are written as an inequality rather than in scientific
    notation, which reads better inside a figure.
    '''
    floor = 10 ** -decimals
    return f'p<{floor:.{decimals}f}' if pval < floor else f'p={pval:.{decimals}f}'


def attach_annotations(results, study='psilodep2', columns=('Condition', 'QIDS_Before')):
    '''
    Adds annotation columns to a results frame, matching on subject_id.

    Models trained without the clinical inputs (no_clinical_features, linreg_on_z, ...)
    write prediction CSVs that carry neither QIDS_Before nor Condition, but the partial
    and within-arm analyses need both for every model.
    '''
    from experiments.ingredients.data_ingredient import add_drug_condition_to_outputs
    from datasets import get_default_prefilter
    from utils.annotations import load_annotations

    if 'Condition' in columns and 'Condition' not in results.columns:
        results = add_drug_condition_to_outputs(results.copy(), study)

    missing = [c for c in columns if c != 'Condition' and c not in results.columns]
    if missing:
        annotations = load_annotations(study=study, filter=get_default_prefilter(study))
        lookup = annotations[['Patient'] + missing].copy()
        lookup['subject_id'] = lookup['Patient'] - 1
        results = results.merge(lookup[['subject_id'] + missing],
                                on='subject_id', how='left')
    return results


# Baseline severity of the two cohorts ---------------------------------------------------

# Both groups received psilocybin: the psilodep2 escitalopram arm is dropped, and
# psilodep1 has no second arm.
BASELINE_GROUPS = [
    {'study': 'psilodep2', 'label': 'psilodep2\n(psilocybin arm)', 'condition': 1},
    {'study': 'psilodep1', 'label': 'psilodep1\n(all psilocybin)', 'condition': None},
]
BASELINE_MEASURES = [('QIDS', 'QIDS_Before'), ('BDI', 'BDI_Before')]

JITTER_SD = 0.06
MARKER_SIZE = 28
SIG_ALPHA = 0.05


def study_annotations(study):
    '''
    Annotations of the patients a study contributes to the models.

    The filter is datasets.get_default_prefilter(), the same one BrainGraphDataset applies,
    so the rows describe exactly the patients that are trained and tested on.
    '''
    from datasets import get_default_prefilter
    from utils.annotations import load_annotations
    return load_annotations(study, filter=get_default_prefilter(study))


def _baseline_samples(group, column):
    '''Baseline scores of one study group, dropping patients without a score.'''
    df = study_annotations(group['study'])
    if group['condition'] is not None:
        df = df[df['Condition_numeric'] == group['condition']]
    return df[column].dropna().to_numpy(dtype=float)


def _compare_studies(measure, column, groups, samples):
    '''
    Welch's t-test and Cohen's d between the baseline scores of two independent groups.

    Welch rather than Student, because the two studies differ in sample size and the
    equality of their variances is not established.
    '''
    a, b = samples
    t, pval = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    return {'measure': measure, 'column': column,
            'group1': groups[0]['study'], 'n1': len(a),
            'mean1': a.mean(), 'std1': a.std(ddof=1),
            'group2': groups[1]['study'], 'n2': len(b),
            'mean2': b.mean(), 'std2': b.std(ddof=1),
            't': t, 'p': pval, 'cohen_d': (a.mean() - b.mean()) / pooled_sd}


def _baseline_group_panel(ax, samples, labels, ylabel, rng):
    '''Boxplots of each group's baseline scores, with the individual patients on top.'''
    scores = pd.DataFrame({'score': np.concatenate(samples),
                           'group': np.repeat(labels, [len(s) for s in samples])})
    sns.boxplot(data=scores, x='group', y='score', order=labels,
                color=BOX_COLOR, width=0.5, showfliers=False, ax=ax, zorder=1)

    # Every patient shown here received psilocybin, hence a single marker style.
    for position, values in enumerate(samples):
        x_jitter = position + rng.normal(0, JITTER_SD, size=len(values))
        ax.scatter(x_jitter, values, marker='d', color=PSILO, edgecolor=PSILO,
                   s=MARKER_SIZE, alpha=ALPHA_SCATTER, zorder=2)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)


def _add_significance_marker(ax, pval, positions=(0., 1.)):
    '''Marks a significant group difference with an asterisk centred above the boxes.'''
    if pval >= SIG_ALPHA:
        return
    ax.text(np.mean(positions), 0.97, '*', transform=ax.get_xaxis_transform(),
            color='red', fontsize=16, fontweight='bold', ha='center', va='top')


def baseline_severity_panels(out, name, rng, groups=None, measures=None):
    '''
    One boxplot per severity measure, comparing the baseline scores of the two cohorts.

    Parameters:
    ----------
        out (FigureOutput): Output handle of the calling target.
        name (str): Panel filename without extension.
        rng (np.random.Generator): Source of the marker jitter.

    Returns:
    -------
        pd.DataFrame: One row per measure, with the group means and Welch's t-test.
    '''
    groups = list(BASELINE_GROUPS if groups is None else groups)
    measures = list(BASELINE_MEASURES if measures is None else measures)

    fig, axes = plt.subplots(1, len(measures), figsize=(3 * len(measures), 3.5))
    axes = np.atleast_1d(axes)

    rows = []
    for ax, (measure, column) in zip(axes, measures):
        samples = [_baseline_samples(group, column) for group in groups]
        labels = [f"{group['label']}\nn={len(sample)}"
                  for group, sample in zip(groups, samples)]
        _baseline_group_panel(ax, samples, labels, f'{measure} Score', rng)

        test = _compare_studies(measure, column, groups, samples)
        _add_significance_marker(ax, test['p'])
        ax.set_title(f"{measure}: t={test['t']:.2f}, {fmt_p_floor(test['p'])}, "
                     f"d={test['cohen_d']:.2f}")
        rows.append(test)

    plt.tight_layout()
    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return pd.DataFrame(rows)


def prediction_metrics(results, label, xcol='label', ycol='prediction'):
    '''
    Accuracy of the mean-across-seed predictions, as one row.
    '''
    r, p = pearsonr(results[xcol], results[ycol])
    err = results[xcol] - results[ycol]
    ss_tot = ((results[xcol] - results[xcol].mean()) ** 2).sum()
    return {'model': label, 'n': len(results), 'r': r, 'p': p,
            'R2': 1 - (err ** 2).sum() / ss_tot,
            'mae': np.abs(err).mean(), 'rmse': np.sqrt((err ** 2).mean()),
            'mae_std': np.abs(err).std(ddof=0)}


def _fisher_z(r1, n1, r2, n2):
    '''Two-sided test that two independent correlations differ.'''
    z = (np.arctanh(r2) - np.arctanh(r1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return z, 2 * norm.sf(abs(z))


def within_arm_metrics(results, label, xcol='label', ycol='prediction'):
    '''
    Accuracy within each treatment arm, plus a test that the two arms differ.
    '''
    per_arm = compute_within_group_pearsonr(
        results, cond_dict=ARM_NAMES, grouping_col='Condition').set_index('Condition')

    rows = []
    for value, arm in ARM_NAMES.items():
        if arm not in per_arm.index:
            continue
        subset = results[results['Condition'] == value]
        abs_err = np.abs(subset[xcol] - subset[ycol])
        rows.append({'model': label, 'arm': arm, 'n': int(per_arm.loc[arm, 'n']),
                     'r': per_arm.loc[arm, 'r'], 'p': per_arm.loc[arm, 'p'],
                     'z': np.nan, 'mae': abs_err.mean(),
                     'bias': (subset[ycol] - subset[xcol]).mean()})

    if len(rows) == 2:
        z, p = _fisher_z(rows[0]['r'], rows[0]['n'], rows[1]['r'], rows[1]['n'])
        rows.append({'model': label, 'arm': 'difference',
                     'n': rows[0]['n'] + rows[1]['n'], 'r': np.nan, 'p': p, 'z': z,
                     'mae': np.nan, 'bias': np.nan})

    return pd.DataFrame(rows)


def report_prediction_metrics(specs, out, name='prediction_metrics', study='psilodep2',
                              heading='Prediction accuracy of the mean predictions',
                              targets=None):
    '''
    Writes the accuracy of several models as CSVs, instead of only into panel titles.

    Parameters:
    ----------
        specs (list): (label, results_file) pairs. Missing directories raise MissingInput,
                      as elsewhere, so a target fails loudly rather than reporting a gap.
        name (str): Basename of the overall table; the within-arm table appends
                    '_within_arm'.
        targets (dict): {label: target name}. Adds a target column, so that the absolute
                        errors of models trained on different outcomes are not read as
                        being in the same units.

    Returns:
    -------
        tuple: (overall_df, within_arm_df)
    '''
    overall, per_arm = [], []
    for label, results_file in specs:
        require(os.path.dirname(results_file))
        results = attach_annotations(
            aggregate_prediction_results(results_file=results_file), study)
        row = prediction_metrics(results, label)
        if targets is not None:
            row = {'model': label, 'target': targets.get(label, ''), **row}
        overall.append(row)
        per_arm.append(within_arm_metrics(results, label))

    overall_df = pd.DataFrame(overall)
    within_arm_df = pd.concat(per_arm, ignore_index=True)

    out.log_df(heading, overall_df)
    out.log_df('Within-arm accuracy, and Fisher z that the arms differ', within_arm_df)
    out.table(name, overall_df)
    out.table(f'{name}_within_arm', within_arm_df)
    return overall_df, within_arm_df


# Partial correlations (notebook cells 22, 24, 25) ---------------------------------------

INTERACTION_COL = 'Condition:QIDS_Before'

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
    {'name': 'Condition + QIDS_Before + interaction',
     'covariates': ['Condition', 'QIDS_Before', INTERACTION_COL],
     'suffix': 'condition_qids_before_interaction'},
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
    Runs the partial-correlation analyses of prediction versus label and plots each.

    Controls for treatment condition, baseline QIDS, both, and both plus their
    interaction.
    '''
    summary_rows = []

    results = results.copy()
    if INTERACTION_COL not in results.columns:
        results[INTERACTION_COL] = results['Condition'] * results['QIDS_Before']

    for analysis in PARTIAL_CORR_ANALYSES:
        covs = analysis['covariates']
        r, p, y_res, x_res = partial_correlation(results, ycol, xcol, covs)

        out.log(f"Partial correlation ({analysis['name']}):")
        out.log(f"   covariates: {', '.join(covs)}")
        out.log(f"   r = {r:.4f}")
        out.log(f"   p = {fmt_p(p)}")
        out.log()

        x_resid = f"{xcol}_controlled_{analysis['suffix']}"
        y_resid = f"{ycol}_controlled_{analysis['suffix']}"

        plot_df = results.copy()
        plot_df[x_resid] = y_res
        plot_df[y_resid] = x_res

        # Need to pass partial-correlation statistics explicitly, otherwise true_vs_pred_scatter would compute standard Pearson test
        abs_err = np.abs(plot_df[x_resid] - plot_df[y_resid])
        true_vs_pred_scatter(plot_df, xcol=x_resid, ycol=y_resid, yerr=yerr,
                             stats={'r': r, 'p': p, 'mae': abs_err.mean(),
                                    'mae_std': abs_err.std(ddof=0)},
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


# Whether a larger value is the better model, per metric. mse is left out: it is a
# monotone transform of rmse, so the rank tests below return identical p-values.
SEED_METRICS = {'r': 'higher', 'r2': 'higher', 'mae': 'lower', 'rmse': 'lower'}

DEFAULT_PREDICTION_FILE = 'prediction_results.csv'


def collect_seed_metric_table(specs, prediction_file=DEFAULT_PREDICTION_FILE):
    '''
    Per-seed r, r2, mae, mse and rmse for several models, as one long dataframe.

    final_metrics.csv carries only r and mae, so the remaining metrics are recomputed
    from each seed's predictions by summarise_seed_metrics, which caches them in the
    model directory. 

    Parameters:
    ----------
        specs (list): (label, directory) pairs, each directory holding seed_*/ subdirs.

    Returns:
    -------
        pd.DataFrame: One row per model and seed, with a 'model' column.
    '''
    summary_file = ('seed_metrics_summary.csv'
                    if prediction_file == DEFAULT_PREDICTION_FILE
                    else f'seed_metrics_{os.path.splitext(prediction_file)[0]}.csv')

    frames = []
    for label, base_dir in specs:
        require(base_dir)
        metrics = summarise_seed_metrics(base_dir=base_dir,
                                         prediction_file=prediction_file,
                                         summary_file=summary_file).copy()
        metrics['seed'] = metrics['seed'].astype(str).str.replace('seed_', '').astype(int)
        metrics['model'] = label
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def compare_across_metrics(seed_table, order, model_of_interest, metrics=None):
    """
    Friedman and pairwise Wilcoxon over several performance metrics, as one long table.

    The r comparison says whether an ablation loses rank-order accuracy; mae and rmse say
    whether it loses calibrated accuracy, and the two need not agree. Each metric is
    corrected as its own BH family, since the metrics are alternative readings of the
    same runs rather than separate hypotheses.

    Parameters:
    ----------
        seed_table (pd.DataFrame): Output of collect_seed_metric_table.
        order (list): Model labels to compare, in panel order.
        model_of_interest (str): Label of the full model each ablation is tested against.
        metrics (list): Metric columns to test. Defaults to SEED_METRICS.

    Returns:
    -------
        pd.DataFrame: One row per test, with columns
                      metric, test, model_a, model_b, n, statistic, p, p_bh, significant.
    """
    metrics = list(SEED_METRICS if metrics is None else metrics)
    rows = []

    for metric in metrics:
        # Seeds are the paired unit, so a seed missing for any model is dropped for all.
        per_seed = seed_table.pivot(
            index='seed', columns='model', values=metric)[order].dropna()
        distributions = {label: per_seed[label].tolist() for label in order}

        global_df, pairwise_df = compare_model_performances(
            distributions, is_dependent=True, model_of_interest=model_of_interest)

        rows.append({'metric': metric, 'test': 'friedman', 'model_a': '(all)',
                     'model_b': '', 'n': len(per_seed),
                     'statistic': global_df['Statistic'].iloc[0],
                     'p': global_df['P-Value'].iloc[0], 'p_bh': np.nan,
                     'significant': bool(global_df['Significant (alpha=0.05)'].iloc[0])})

        for _, row in pairwise_df.iterrows():
            rows.append({
                'metric': metric, 'test': 'wilcoxon',
                'model_a': row['Model A'], 'model_b': row['Model B'], 'n': len(per_seed),
                'statistic': np.nan, 'p': row['Original P-Value'],
                'p_bh': row['Corrected P-Value'],
                'significant': bool(row['Reject Null (Significant Difference)'])})

    return pd.DataFrame(rows)


def model_comparison_panels(specs, out, name, num_subs, model_of_interest,
                            sort_by_mean=True, offset=4, figsize=(8, 4),
                            skip_missing=False, table_prefix=''):
    '''Runs the full seed-sensitivity block: collect metrics, raincloud, comparison stats.'''
    metrics_df = collect_seed_metrics(specs, skip_missing=skip_missing)
    distributions = metrics_to_distributions(metrics_df, sort_by_mean=sort_by_mean)
    raincloud_of_model_r(distributions, out, name, num_subs, offset=offset, figsize=figsize)
    report_model_comparison(distributions, out, model_of_interest, table_prefix=table_prefix)
    return distributions


# Permutation importance ------------------------------------------------------------------

def importance_panel(importance_dir, weights_dir, out, name='importance_scores_aggregated',
                     metrics_file='final_metrics.csv', **kwargs):
    '''
    Bar chart of the aggregated importance scores, annotated with the relative error increase.

    Each score is the increase in mean absolute error when a block of MLP inputs is
    shuffled, measured against one seed's out-of-fold predictions.

    Parameters:
    ----------
        importance_dir (str): Directory holding importance_scores_aggregated.csv.
        weights_dir (str): Directory of the seed_*/ runs the importances were computed on.
        name (str): Panel filename, and the stem of the tables written beside it.

    Returns:
    -------
        tuple: (scores, baseline_df)
    '''
    scores = aggregate_importance_scores(
        os.path.join(importance_dir, 'importance_scores_aggregated.csv'))
    scores = scores.sort_values(by='mean', ascending=False)

    seed_mae = collect_seed_metrics([(name, weights_dir, metrics_file)])['mae']
    baseline = seed_mae.mean()
    scores['percent_of_baseline'] = 100 * scores['mean'] / baseline

    permutation_importance_bar_chart(scores, yerr_column='se', color=NEUTRAL, alpha=0.8,
                                     baseline=baseline, save_path=out.fig(name), **kwargs)

    baseline_df = pd.DataFrame([{'n_seeds': len(seed_mae),
                                 'baseline_mae': baseline,
                                 'baseline_mae_sem': seed_mae.sem()}])
    out.table(name, scores)
    out.table(f'{name}_baseline', baseline_df)
    out.log(f'Permutation importance baseline: MAE = {baseline:.3f} +/- {seed_mae.sem():.3f} '
            f'(mean +/- sem across {len(seed_mae)} seeds)')
    return scores, baseline_df


# Feature ablations ----------------------------------------------------------------------

# Column order of the domain-presence matrix.
INPUT_DOMAINS = ['Clinical', 'FC', 'REACT']

# Names of the ablated models in the figures, keyed by the results directory they live in.
# Shared so that the psilodep2 ablation panels and the psilodep1 transfer panels label the
# same model identically.
ABLATION_NAMES = {
    'no_clinical_features': 'FC + REACT',
    'no_node_features': 'FC + clinical',
    'no_react_no_clinical': 'FC only',
}


def _domain_matrix(ax, labels, domains, domain_names):
    '''
    Draws the filled/empty markers that say which input domains each model received.

    Reading the rows as a matrix is what makes the ablation a design rather than a list
    of models, which is the comparison the panel exists to support.
    '''
    for row, label in enumerate(labels):
        present = domains.get(label, ())
        ax.plot([0, len(domain_names) - 1], [row, row],
                color=NEUTRAL, alpha=0.15, linewidth=1, zorder=1)
        for col, domain in enumerate(domain_names):
            ax.scatter(col, row, s=70, zorder=3, linewidth=1.2, edgecolor=NEUTRAL,
                       facecolor=NEUTRAL if domain in present else 'white')

    ax.set_xticks(range(len(domain_names)))
    ax.set_xticklabels(domain_names, rotation=45, ha='right')
    ax.set_xlim(-0.6, len(domain_names) - 0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def feature_ablation_panel(distributions, domains, out, name,
                           domain_names=None, figsize=(9, 4), offset=4):
    '''
    Raincloud of the feature ablations, with a domain-presence matrix down the left.

    Takes the per-seed values rather than a directory, so that the panel and the tests
    beside it are guaranteed to be reading the same numbers.

    Parameters:
    ----------
        distributions (dict): {label: r per seed}, ordered top to bottom.
        domains (dict): {label: iterable of domain names the model received}.
        out (FigureOutput): Output handle of the calling target.
        name (str): Panel filename without extension.
        domain_names (list): Column order of the matrix. Defaults to INPUT_DOMAINS.
    '''
    domain_names = list(INPUT_DOMAINS if domain_names is None else domain_names)

    # plot_raincloud places the first entry at the bottom, so reverse to make the order
    # of `specs` read top to bottom.
    plotted = dict(reversed(list(distributions.items())))

    colors = plot_colormap_stack('YlGnBu', len(plotted) + offset, make_plot=False)[offset:]
    palette = {label: color for label, color in zip(plotted.keys(), colors)}

    fig, (ax_matrix, ax_rain) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw={'width_ratios': [0.45 * len(domain_names), 4], 'wspace': 0.06})

    plot_raincloud(plotted, palette=palette, ax=ax_rain, alpha=0.5, box_alpha=0.3,
                   sort_by_mean=False)
    ax_rain.set_xlabel('r')
    ax_rain.tick_params(labelleft=False)

    _domain_matrix(ax_matrix, list(plotted.keys()), domains, domain_names)

    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


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
