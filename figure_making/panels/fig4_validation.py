"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

graphTRIP is applied zero-shot: no weight is updated and no psilodep1 outcome is used.

Panels:
- b. baseline severity of the two cohorts, which the model is asked to extrapolate across
- c. zero-shot prediction of graphTRIP
- d. reconstruction performance of the graphTRIP VGAE, not fine-tuned, tested against
     the primary-dataset reconstructions of Fig. 2d
- e. r and partial r, for the full model, the imaging-only ablation and the clinical-only
     benchmark. One bar panel removes baseline QIDS from both sides, a second removes
     baseline QIDS and BDI, the two severity scores graphTRIP receives

Panels c and e are drawn without clinical harmonisation for the main figure, and again
with it as separate files.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils.configs import load_ingredient_configs
from utils.helpers import aggregate_prediction_results
from utils.plotting import NEUTRAL, NEUTRAL2, PSILO, regression_scatter
from utils.statsalg import compare_reconstruction_performance

from figure_making.common import (
    ABLATION_NAMES, attach_annotations, baseline_severity_panels, partial_correlation,
    plot_correlation_boxplot)
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, perm_dirs, require
from figure_making.registry import register


# Independent fold assignments drawn for the ensemble-matched reconstruction control
N_MATCHED_DRAWS = 10

# (label, filename suffix) of the two input mappings. The first is the main figure's.
CONDITIONS = [('no harmonisation', ''), ('harmonised', '_harmonised')]

# Zero-shot predictions of one model, averaged over its fold models.
PREDICTIONS_FILE = 'initial_prediction_results_mean_vote'

GRAPHTRIP = 'Full graphTRIP'
NO_CLINICAL = 'graphTRIP without clinical data'
CLINICAL_ONLY = 'Clinical-only MLP'

# (results directory parts, label) of every model in the summary table. control_mlp_raw is
# the clinical-only MLP retrained for the transfer; the stale control_mlp/ tree is not read.
MODELS = [
    (('validation', 'evaluate_graphtrip'), GRAPHTRIP),
    (('validation', 'feature_ablation', 'no_node_features'),
     ABLATION_NAMES['no_node_features']),
    (('validation', 'feature_ablation', 'no_clinical_features'), NO_CLINICAL),
    (('validation', 'feature_ablation', 'no_react_no_clinical'),
     ABLATION_NAMES['no_react_no_clinical']),
    (('validation', 'feature_ablation', 'control_mlp_raw'), CLINICAL_ONLY),
]

# Subset shown in the bar panel; the benchmark is pinned to the bottom.
PANEL_MODELS = [GRAPHTRIP, NO_CLINICAL, CLINICAL_ONLY]

# Null models exist for graphTRIP only, so its p-values are NaN for every other model.
NULL_MODEL = GRAPHTRIP
NULL_PARTS = ('graphtrip', 'permutation_null')

# Predictors compared in the sensitivity tables. QIDS_Before is the score itself, used as
# a zero-parameter predictor, so it has no partial correlation given itself.
SENSITIVITY_MODELS = [GRAPHTRIP, CLINICAL_ONLY]
BASELINE_PREDICTOR = 'QIDS_Before'

# The baseline severity scores graphTRIP reads, attached to every model's predictions.
BASELINE_COVARIATES = ('QIDS_Before', 'BDI_Before')

# Covariate sets removed from both sides of the correlation, one bar panel each.
PARTIAL_SPECS = [
    {'suffix': '', 'covariates': ['QIDS_Before'], 'legend': 'partial r | QIDS_Before'},
    {'suffix': '_qids_bdi', 'covariates': list(BASELINE_COVARIATES),
     'legend': 'partial r | QIDS_Before, BDI_Before'},
]
PARTIAL_COLUMNS = [f"partial_r{spec['suffix']}" for spec in PARTIAL_SPECS]

SCATTER_XLABEL = 'True QIDS, 1 week post 10+25-mg psilocybin for TRD'
SCATTER_YLABEL = 'Predicted QIDS, 3 weeks post 2x25-mg psilocybin for MDD'

BAR_COLOR_FULL = NEUTRAL
BAR_COLOR_PARTIAL = PSILO
BAR_HEIGHT = 0.68
NESTED_FRACTION = 0.42


def matched_fold_assignment(num_subs, num_folds, num_seeds, rng):
    '''
    Draws a random k-fold assignment for a dataset that no model was trained on.

    Each psilodep2 subject is reconstructed only by the one VGAE per seed that held them
    out, so their reconstruction is an average over num_seeds models. Reconstructing
    psilodep1 with every VGAE averages num_seeds * num_folds models instead, and the
    extra averaging cancels sampling noise. Spreading the psilodep1 subjects over the
    folds of each seed restores the matched ensemble size.

    Returns:
    -------
        dict: {seed key: array of length num_subs holding the fold each subject is
               assigned to}, in the format get_mean_test_reconstructions expects.
    '''
    if num_subs < num_folds:
        raise ValueError(f'Cannot spread {num_subs} subjects over {num_folds} folds: '
                         'every fold needs at least one subject.')
    # Balanced fold sizes, then shuffled independently for each seed
    folds = np.tile(np.arange(num_folds), int(np.ceil(num_subs / num_folds)))[:num_subs]
    return {f'seed_{seed}': rng.permutation(folds) for seed in range(num_seeds)}


def summarise_matched_draws(per_draw):
    '''
    Condenses the per-draw comparisons of the ensemble-matched control into one row per
    feature. Each draw is a different arbitrary assignment of val subjects to folds.
    '''
    df = pd.concat(per_draw, ignore_index=True)
    summary = df.groupby('feature', sort=False).agg(
        n_draws=('feature', 'size'),
        mean_primary=('mean_primary', 'first'),
        median_mean_validation=('mean_validation', 'median'),
        median_mean_difference=('mean_difference', 'median'),
        median_cohens_d=('cohens_d', 'median'),
        median_cohens_d_ci_low=('cohens_d_ci_low', 'median'),
        median_cohens_d_ci_high=('cohens_d_ci_high', 'median'),
        median_p_uncorrected=('p_uncorrected', 'median'),
        median_p_fdr=('p_fdr', 'median'),
        frac_draws_significant_fdr=('significant_fdr', 'mean'))
    return summary.reset_index()


def reconstruction_comparison(ctx, out, psilodep1_data):
    '''
    Tests whether VGAE reconstruction quality differs between the primary dataset
    (Fig. 2d) and the independent validation dataset (Fig. 4d).
    '''
    _, primary_x = ctx.core_reconstructions
    _, psilodep1_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, None)

    num_folds = len(ctx.vgaes_dict['seed_0'])
    out.log('=== Primary vs validation reconstruction ===')
    out.log(f'Panel ensembles: psilodep2 averages {ctx.num_seeds} VGAEs per subject '
            f'(the held-out fold of each seed), psilodep1 averages '
            f'{ctx.num_seeds * num_folds} (every fold of every seed).')

    test_results = compare_reconstruction_performance(primary_x['metrics'],
                                                      psilodep1_x['metrics'])
    out.table('primary_vs_validation_corr', test_results['corr'])
    out.table('primary_vs_validation_mae', test_results['mae'])
    out.log_df('Primary vs validation reconstruction (correlation)', test_results['corr'])
    out.log_df('Primary vs validation reconstruction (MAE)', test_results['mae'])

    # Ensemble-matched control: one VGAE per seed for the validation dataset too
    rng = np.random.default_rng(ctx.cfg.seed)
    per_draw = {'corr': [], 'mae': []}
    for draw in range(N_MATCHED_DRAWS):
        assignment = matched_fold_assignment(len(psilodep1_data), num_folds,
                                             ctx.num_seeds, rng)
        _, matched_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, assignment)
        for metric, df in compare_reconstruction_performance(primary_x['metrics'],
                                                             matched_x['metrics']).items():
            per_draw[metric].append(df.assign(draw=draw))

    out.log(f'Ensemble-matched control: each psilodep1 subject reconstructed by one '
            f'randomly drawn fold per seed instead of all {num_folds}, so that subjects '
            f'of both datasets average {ctx.num_seeds} VGAEs; {N_MATCHED_DRAWS} draws.')
    for metric, label in [('corr', 'correlation'), ('mae', 'MAE')]:
        summary = summarise_matched_draws(per_draw[metric])
        out.table(f'primary_vs_validation_{metric}_ensemble_matched', summary)
        out.log_df(f'Ensemble-matched primary vs validation reconstruction ({label})',
                   summary)

    return psilodep1_x


# Zero-shot predictions -------------------------------------------------------------------

def load_zeroshot_results(base_dir, suffix):
    '''
    Zero-shot predictions of one model under one input mapping, with the baseline scores.

    Models trained without clinical inputs read no severity score, so harmonisation is the
    identity for them and only the unharmonised file was written; the harmonised condition
    falls back to it. Their prediction CSVs also carry no baseline scores, which the partial
    correlations need, so they are joined on from the psilodep1 annotations.
    '''
    path = os.path.join(base_dir, f'{PREDICTIONS_FILE}{suffix}.csv')
    if not os.path.exists(path):
        path = require(os.path.join(base_dir, f'{PREDICTIONS_FILE}.csv'))
    return attach_annotations(pd.read_csv(path), study='psilodep1',
                              columns=BASELINE_COVARIATES)


def correlation_row(results, ycol='prediction'):
    '''
    The correlations one bar summarises.

    r is the raw agreement with the outcome; each partial_r is what survives after one
    covariate set is removed from both sides; r_with_<score> says how much of the
    prediction is that baseline score. A partial correlation is undefined when the
    predictor is one of its own covariates.
    '''
    row = {'n': len(results),
           'r': results[ycol].corr(results['label']),
           **{f'r_with_{c}': results[ycol].corr(results[c])
              for c in BASELINE_COVARIATES}}
    for column, spec in zip(PARTIAL_COLUMNS, PARTIAL_SPECS):
        row[column] = (np.nan if ycol in spec['covariates'] else
                       partial_correlation(results, ycol, 'label', spec['covariates'])[0])
    return row


def seed_rows(base_dir, suffix):
    '''The same correlations computed within each training seed, one row per seed.'''
    rows = []
    # Filtered to directories: the cached seed_metrics_*.csv summaries sit alongside them.
    for seed_dir in sorted(d for d in glob.glob(os.path.join(base_dir, 'seed_*'))
                           if os.path.isdir(d)):
        path = os.path.join(seed_dir, f'{PREDICTIONS_FILE}{suffix}.csv')
        if not os.path.exists(path):
            path = os.path.join(seed_dir, f'{PREDICTIONS_FILE}.csv')
        if not os.path.exists(path):
            continue
        results = attach_annotations(pd.read_csv(path), study='psilodep1',
                                     columns=BASELINE_COVARIATES)
        rows.append(correlation_row(results))
    return pd.DataFrame(rows)


def null_draws(reference, suffix, num_seeds):
    '''
    Correlations of the null models, one draw per permutation.

    scripts/permutation_null.py retrains graphTRIP on permuted psilodep2 outcomes and
    transfers each null model to psilodep1, so a draw is what this pipeline produces when
    nothing was learned from the primary cohort. Each draw averages one permutation's fold
    models exactly as the observed value does, and is scored against the true psilodep1
    outcomes joined on by subject_id. Permutations with an incomplete seed set are dropped.

    Returns None when the null tree is absent, so the panels still build without it.
    '''
    base = output_dir(*NULL_PARTS)
    if not os.path.exists(base):
        return None

    truth = reference.set_index('subject_id')[
        ['label', *BASELINE_COVARIATES]].sort_index()
    draws = []
    for perm_dir in perm_dirs(base):
        paths = sorted(glob.glob(os.path.join(perm_dir, 'psilodep1', 'seed_*',
                                              f'{PREDICTIONS_FILE}{suffix}.csv')))
        if len(paths) != num_seeds:
            continue
        predictions = pd.concat(
            [pd.read_csv(p).set_index('subject_id')['prediction'] for p in paths],
            axis=1).mean(axis=1)
        draws.append(correlation_row(
            truth.assign(prediction=predictions.reindex(truth.index)).reset_index()))
    return pd.DataFrame(draws) if draws else None


def null_pvalues(draws, row,
                 statistics=('r', *PARTIAL_COLUMNS,
                             *[f'r_with_{c}' for c in BASELINE_COVARIATES])):
    '''Two-sided rank p of each observed correlation against the null draws.'''
    if draws is None:
        return {f'{statistic}_p': np.nan for statistic in statistics}
    return {f'{statistic}_p':
            (1 + (draws[statistic].abs() >= abs(row[statistic])).sum()) / (1 + len(draws))
            for statistic in statistics}


def correlation_table(out, num_seeds):
    '''
    One row per model and input mapping: every correlation, and the p-values of the model
    the null was run for.

    Returns:
    -------
        tuple: (table, {(condition label, 'ensemble' | 'seeds'): frame}) where each frame
               is ordered by ensemble r with the benchmark pinned last.
    '''
    rows, panels = [], {}

    for condition, suffix in CONDITIONS:
        reference = load_zeroshot_results(
            require(output_dir('validation', 'evaluate_graphtrip')), suffix)
        draws = null_draws(reference, suffix, num_seeds)
        if draws is None:
            out.log(f'Null models not found under {output_dir(*NULL_PARTS)}; '
                    f'those p-values are left NaN.')

        ensemble, seeds = [], []
        for parts, label in MODELS:
            results = load_zeroshot_results(require(output_dir(*parts)), suffix)
            row = correlation_row(results)
            per_seed = seed_rows(output_dir(*parts), suffix)

            ensemble.append({'model': label, 'r': row['r'],
                             **{c: row[c] for c in PARTIAL_COLUMNS}})
            seeds.append({'model': label, 'r': per_seed['r'].mean(),
                          'r_err': per_seed['r'].std(ddof=1),
                          **{k: v for c in PARTIAL_COLUMNS
                             for k, v in ((c, per_seed[c].mean()),
                                          (f'{c}_err', per_seed[c].std(ddof=1)))}})

            rows.append({
                'model': label, 'condition': condition, 'n': row['n'],
                'n_seeds': len(per_seed),
                **{k: row[k] for k in ('r', *PARTIAL_COLUMNS,
                                       *[f'r_with_{c}' for c in BASELINE_COVARIATES])},
                **null_pvalues(draws if label == NULL_MODEL else None, row),
                'seed_mean_r': per_seed['r'].mean(), 'seed_sd_r': per_seed['r'].std(ddof=1),
                **{k: v for c in PARTIAL_COLUMNS
                   for k, v in ((f'seed_mean_{c}', per_seed[c].mean()),
                                (f'seed_sd_{c}', per_seed[c].std(ddof=1)))}})

        order = order_by_r(pd.DataFrame(ensemble))['model'].tolist()
        panels[(condition, 'ensemble')] = reorder(pd.DataFrame(ensemble), order)
        panels[(condition, 'seeds')] = reorder(pd.DataFrame(seeds), order)

    return pd.DataFrame(rows), panels


def order_by_r(frame):
    '''Highest r on top, with the clinical-only benchmark pinned to the bottom.'''
    pinned = frame[frame['model'] == CLINICAL_ONLY]
    rest = frame[frame['model'] != CLINICAL_ONLY].sort_values('r', ascending=False)
    return pd.concat([rest, pinned], ignore_index=True)


def reorder(frame, order):
    '''Puts frame's rows in the given model order.'''
    return frame.set_index('model').loc[order].reset_index()


def subset(frame, models):
    '''Keeps the given models, in the order the frame already has.'''
    return frame[frame['model'].isin(models)].reset_index(drop=True)


def zeroshot_scatter(results, out, name, pvalue=None):
    '''
    Zero-shot predictions against the observed outcome, with a fitted regression line.

    No identity line: the two cohorts differ in treatment protocol, patient population
    and outcome timepoint, so the predictions are not expected to be calibrated to the
    validation scale, only to covary with it. The axes are square but their limits are
    independent, for the same reason.
    '''
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    regression_scatter(results, xcol='label', ycol='prediction', yerr='prediction_sem',
                       show_ci=False, ax=ax)

    ax.set_box_aspect(1)
    ax.set_xlabel(SCATTER_XLABEL, fontsize=9)
    ax.set_ylabel(SCATTER_YLABEL, fontsize=9)

    title = f"r = {results['prediction'].corr(results['label']):.4f}"
    if pvalue is not None and not np.isnan(pvalue):
        title += f', p = {pvalue:.4f}'
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def correlation_bar_panel(out, frame, name, partial_col='partial_r',
                          partial_label='partial r | QIDS_Before'):
    '''
    One horizontal bar per model: r as the full bar, partial r nested inside it.

    The nested bar is drawn narrower and in front rather than stacked, because partial r
    is not a component of r. 
    '''
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    positions = np.arange(len(frame))[::-1]

    ax.barh(positions, frame['r'], height=BAR_HEIGHT, color=BAR_COLOR_FULL,
            edgecolor=NEUTRAL2, linewidth=0.6, zorder=2,
            label='r (prediction, QIDS_1week)')
    ax.barh(positions, frame[partial_col], height=BAR_HEIGHT * NESTED_FRACTION,
            color=BAR_COLOR_PARTIAL, zorder=3, label=partial_label)

    ax.axvline(0, color=NEUTRAL2, linewidth=0.8, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame['model'])
    ax.set_xlabel('correlation with QIDS_1week')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize='small', loc='lower center',
              bbox_to_anchor=(0.5, -0.55), ncol=2)

    fig.tight_layout()
    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# Sensitivity of the reported correlations ------------------------------------------------

def jackknife_range(results, ycol):
    '''
    Recomputes r and every partial r on each leave-one-patient-out subset.

    Returns:
    -------
        dict: min, max and sign count of each statistic over the n subsets.
    '''
    values = {statistic: [] for statistic in ('r', *PARTIAL_COLUMNS)}
    for drop in range(len(results)):
        subset = results.drop(results.index[drop])
        row = correlation_row(subset, ycol=ycol)
        for statistic in values:
            values[statistic].append(row[statistic])

    summary = {'n_subsets': len(results)}
    for statistic, series in values.items():
        series = np.asarray(series, dtype=float)
        if np.isnan(series).all():
            summary.update({f'{statistic}_jk_min': np.nan, f'{statistic}_jk_max': np.nan,
                            f'{statistic}_jk_positive': np.nan})
            continue
        summary.update({f'{statistic}_jk_min': series.min(),
                        f'{statistic}_jk_max': series.max(),
                        f'{statistic}_jk_positive': int((series > 0).sum())})
    return summary


def rank_statistics(results, ycol):
    '''Spearman rho of the predictor with the outcome, raw and given each covariate set.'''
    rho, pval = stats.spearmanr(results[ycol], results['label'])
    row = {'spearman_rho': rho, 'spearman_p': pval}
    for spec in PARTIAL_SPECS:
        column = f"partial_spearman_rho{spec['suffix']}"
        if ycol in spec['covariates']:
            row[column] = np.nan
            continue
        ranked = pd.DataFrame({c: results[c].rank()
                               for c in (ycol, 'label', *spec['covariates'])})
        row[column] = partial_correlation(ranked, ycol, 'label', spec['covariates'])[0]
    return row


def sensitivity_table(out):
    '''
    Leave-one-patient-out ranges and rank correlations for the reported predictors.

    Covers the two models in the bar panel plus QIDS_Before used directly as a
    zero-parameter predictor, which is condition-independent and therefore reported once.
    '''
    directories = {label: output_dir(*parts) for parts, label in MODELS}
    rows = []

    for condition, suffix in CONDITIONS:
        for label in SENSITIVITY_MODELS:
            results = load_zeroshot_results(require(directories[label]), suffix)
            rows.append({'model': label, 'condition': condition,
                         **correlation_row(results),
                         **jackknife_range(results, 'prediction'),
                         **rank_statistics(results, 'prediction')})

    results = load_zeroshot_results(
        require(output_dir('validation', 'evaluate_graphtrip')), CONDITIONS[0][1])
    rows.append({'model': BASELINE_PREDICTOR, 'condition': 'n/a',
                 **correlation_row(results, ycol='QIDS_Before'),
                 **jackknife_range(results, 'QIDS_Before'),
                 **rank_statistics(results, 'QIDS_Before')})

    table = pd.DataFrame(rows)
    out.table('zeroshot_sensitivity', table)
    out.log_df('Leave-one-patient-out ranges and rank correlations', table)
    return table


@register('fig4', group='main', subdir='Fig.4')
def fig4_validation(ctx, out):
    results_base_dir = require(output_dir('validation', 'evaluate_graphtrip'))
    main_condition = CONDITIONS[0][0]

    # b. Baseline severity of the two cohorts --------------------------------------------
    baseline_tests = baseline_severity_panels(
        out, 'qids_bdi_baseline_by_study', np.random.default_rng(ctx.cfg.seed))
    out.table('baseline_by_study_tests', baseline_tests)
    out.log_df('Baseline severity, psilodep2 psilocybin arm versus psilodep1 '
               "(Welch's t-tests, uncorrected)", baseline_tests)

    # d. Reconstruction performance on the validation dataset ----------------------------
    psilodep1_config = load_ingredient_configs(os.path.join(results_base_dir, 'seed_0'),
                                               ingredients=['dataset'])
    psilodep1_data = load_dataset(psilodep1_config['dataset'])

    # All psilodep1 patients were treated with psilocybin
    psilodep1_conditions = np.ones(len(psilodep1_data))

    # Every VGAE of every seed and fold reconstructs every patient, then averages, and
    # the result is tested against the primary dataset of Fig. 2d
    psilodep1_x = reconstruction_comparison(ctx, out, psilodep1_data)

    plot_correlation_boxplot(out, psilodep1_x, psilodep1_conditions,
                             'original_vs_reconstructed_corrs')

    # c, e. Zero-shot prediction and the input-domain contrast ---------------------------
    table, panels = correlation_table(out, ctx.num_seeds)
    out.table('zeroshot_correlations', table)
    out.log_df('Zero-shot correlations by model and input mapping', table)

    graphtrip = table[table['model'] == GRAPHTRIP].set_index('condition')
    for condition, suffix in CONDITIONS:
        tag = '' if condition == main_condition else '_harmonised'
        zeroshot_scatter(
            aggregate_prediction_results(results_file=os.path.join(
                results_base_dir, f'{PREDICTIONS_FILE}{suffix}.csv')),
            out, f'zeroshot_true_vs_pred{tag}', graphtrip.loc[condition, 'r_p'])
        frame = subset(panels[(condition, 'ensemble')], PANEL_MODELS)
        for column, spec in zip(PARTIAL_COLUMNS, PARTIAL_SPECS):
            correlation_bar_panel(out, frame,
                                  f"zeroshot_correlation_bars{spec['suffix']}{tag}",
                                  partial_col=column, partial_label=spec['legend'])

    sensitivity_table(out)
