"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

graphTRIP is applied zero-shot: no weight is updated and no psilodep1 outcome is used.

Panels:
- b. baseline severity of the two cohorts, which the model is asked to extrapolate across
- c. zero-shot prediction of graphTRIP, with the baseline severity scores harmonised onto
     the training scale
- d. reconstruction performance of the graphTRIP VGAE, not fine-tuned, tested against
     the primary-dataset reconstructions of Fig. 2d

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.configs import load_ingredient_configs
from utils.helpers import aggregate_prediction_results
from utils.plotting import regression_scatter
from utils.statsalg import compare_reconstruction_performance

from figure_making.common import (
    baseline_severity_panels, plot_correlation_boxplot, study_annotations)
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, perm_dirs, require
from figure_making.registry import register


# Independent fold assignments drawn for the ensemble-matched reconstruction control
N_MATCHED_DRAWS = 10

# (label, filename suffix) of the two input mappings. Panel c shows the harmonised one;
# the spread table reports both.
CONDITIONS = [('no harmonisation', ''), ('harmonised', '_harmonised')]
PANEL_SUFFIX = '_harmonised'

# Zero-shot predictions of one model, averaged over its fold models.
PREDICTIONS_FILE = 'initial_prediction_results_mean_vote'

# graphTRIP retrained on permuted outcomes and transferred to psilodep1
NULL_PARTS = ('graphtrip', 'permutation_null')

# Outcome the models were trained on, and the arm they were trained on.
TRAINING_TARGET = 'QIDS_Final_Integration'
PSILOCYBIN_CONDITION = 1

# Columns shared by the two reconstruction ensembles.
COMPARISON_COLUMNS = ['ensemble', 'feature', 'n_draws', 'n_primary', 'mean_primary',
                      'n_validation', 'mean_validation', 'mean_difference', 'cohens_d',
                      'cohens_d_ci_low', 'cohens_d_ci_high', 'p_uncorrected', 'p_fdr',
                      'frac_significant_fdr']

SCATTER_XLABEL = 'True QIDS, 1 week post 10+25-mg psilocybin for TRD'
SCATTER_YLABEL = 'Predicted QIDS, 3 weeks post 2x25-mg psilocybin for MDD'


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
    feature, taking the median over draws. Each draw is a different arbitrary assignment
    of val subjects to folds.
    '''
    df = pd.concat(per_draw, ignore_index=True)
    summary = df.groupby('feature', sort=False).agg(
        n_draws=('feature', 'size'),
        n_primary=('n_primary', 'first'),
        mean_primary=('mean_primary', 'first'),
        n_validation=('n_validation', 'first'),
        mean_validation=('mean_validation', 'median'),
        mean_difference=('mean_difference', 'median'),
        cohens_d=('cohens_d', 'median'),
        cohens_d_ci_low=('cohens_d_ci_low', 'median'),
        cohens_d_ci_high=('cohens_d_ci_high', 'median'),
        p_uncorrected=('p_uncorrected', 'median'),
        p_fdr=('p_fdr', 'median'),
        frac_significant_fdr=('significant_fdr', 'mean'))
    return summary.reset_index()


def stack_reconstruction_tests(all_folds, matched):
    '''
    The two ensembles in one frame, one row per feature and ensemble.

    all_folds rows are single tests, so n_draws is 1 and frac_significant_fdr is 0 or 1;
    ensemble_matched rows are medians over the draws.
    '''
    all_folds = all_folds.assign(
        ensemble='all_folds', n_draws=1,
        frac_significant_fdr=all_folds['significant_fdr'].astype(float))
    matched = matched.assign(ensemble='ensemble_matched')
    return pd.concat([all_folds[COMPARISON_COLUMNS], matched[COMPARISON_COLUMNS]],
                     ignore_index=True)


def reconstruction_comparison(ctx, out, psilodep1_data):
    '''
    Tests whether VGAE reconstruction quality differs between the primary dataset
    (Fig. 2d) and the independent validation dataset (Fig. 4d).

    Only the correlations are tested, under two ensembles: all_folds is what panel d
    plots, ensemble_matched equalises the number of VGAEs averaged per subject.
    '''
    _, primary_x = ctx.core_reconstructions
    _, psilodep1_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, None)

    num_folds = len(ctx.vgaes_dict['seed_0'])
    out.log('=== Primary vs validation reconstruction ===')
    out.log(f'all_folds: psilodep2 averages {ctx.num_seeds} VGAEs per subject (the '
            f'held-out fold of each seed), psilodep1 averages '
            f'{ctx.num_seeds * num_folds} (every fold of every seed).')
    out.log(f'ensemble_matched: each psilodep1 subject is reconstructed by one randomly '
            f'drawn fold per seed instead of all {num_folds}, so that subjects of both '
            f'datasets average {ctx.num_seeds} VGAEs; {N_MATCHED_DRAWS} draws.')

    all_folds = compare_reconstruction_performance(primary_x['metrics'],
                                                   psilodep1_x['metrics'])['corr']

    rng = np.random.default_rng(ctx.cfg.seed)
    per_draw = []
    for draw in range(N_MATCHED_DRAWS):
        assignment = matched_fold_assignment(len(psilodep1_data), num_folds,
                                             ctx.num_seeds, rng)
        _, matched_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, assignment)
        per_draw.append(compare_reconstruction_performance(
            primary_x['metrics'], matched_x['metrics'])['corr'].assign(draw=draw))

    table = stack_reconstruction_tests(all_folds, summarise_matched_draws(per_draw))
    out.table('d_reconstruction_tests', table)
    out.log_df('Primary vs validation reconstruction (correlation)', table)

    return psilodep1_x


# Zero-shot predictions -------------------------------------------------------------------

def zeroshot_results(base_dir, suffix):
    '''Zero-shot predictions of graphTRIP under one input mapping, averaged over seeds.'''
    return aggregate_prediction_results(
        results_file=os.path.join(base_dir, f'{PREDICTIONS_FILE}{suffix}.csv'))


def null_draws(reference, suffix, num_seeds):
    '''
    Zero-shot r of the null models, one draw per permutation.

    scripts/permutation_null.py retrains graphTRIP on permuted psilodep2 outcomes and
    transfers each null model to psilodep1, so a draw is what this pipeline produces when
    nothing was learned from the primary cohort. Each draw averages one permutation's fold
    models exactly as the observed value does, and is scored against the true psilodep1
    outcomes joined on by subject_id. Permutations with an incomplete seed set are dropped.

    Returns None when the null tree is absent, so the panel still builds without it.
    '''
    base = output_dir(*NULL_PARTS)
    if not os.path.exists(base):
        return None

    truth = reference.set_index('subject_id')['label'].sort_index()
    draws = []
    for perm_dir in perm_dirs(base):
        paths = sorted(glob.glob(os.path.join(perm_dir, 'psilodep1', 'seed_*',
                                              f'{PREDICTIONS_FILE}{suffix}.csv')))
        if len(paths) != num_seeds:
            continue
        predictions = pd.concat(
            [pd.read_csv(p).set_index('subject_id')['prediction'] for p in paths],
            axis=1).mean(axis=1)
        draws.append(predictions.reindex(truth.index).corr(truth))
    return np.array(draws) if draws else None


def zeroshot_pvalue(out, results, num_seeds):
    '''
    Two-sided rank p of the zero-shot r against the null draws, or None without them.
    '''
    r = results['prediction'].corr(results['label'])
    draws = null_draws(results, PANEL_SUFFIX, num_seeds)

    out.log('=== Zero-shot prediction (harmonised) ===')
    if draws is None:
        out.log(f'graphTRIP: r = {r:.4f}, n = {len(results)}. Null models not found under '
                f'{output_dir(*NULL_PARTS)}, so no p-value is reported.')
        return None

    pvalue = (1 + (np.abs(draws) >= abs(r)).sum()) / (1 + len(draws))
    out.log(f'graphTRIP: r = {r:.4f}, n = {len(results)}, p = {pvalue:.4f} '
            f'(two-sided rank p against {len(draws)} null draws)')
    return pvalue


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


def spread_row(values, label, reference_sd):
    '''Spread of one score or prediction, and its SD relative to reference_sd.'''
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    sd = values.std(ddof=1)
    return {'series': label, 'n': len(values), 'mean': values.mean(), 'sd': sd,
            'min': values.min(), 'max': values.max(),
            'range': values.max() - values.min(), 'sd_ratio': sd / reference_sd}


def prediction_spread_table(out):
    '''
    Spread of the training outcome, the validation outcome and the zero-shot predictions.

    sd_ratio is relative to the observed psilodep1 outcome, so it says how much of the
    outcome's spread the predictions reproduce.
    '''
    base_dir = require(output_dir('validation', 'evaluate_graphtrip'))
    predictions = {condition: zeroshot_results(base_dir, suffix)
                   for condition, suffix in CONDITIONS}
    observed = predictions[CONDITIONS[0][0]]['label']
    reference_sd = observed.std(ddof=1)

    training = study_annotations('psilodep2')
    training = training[training['Condition_numeric'] == PSILOCYBIN_CONDITION]

    rows = [spread_row(training[TRAINING_TARGET],
                       f'psilodep2 {TRAINING_TARGET} (psilocybin arm)', reference_sd),
            spread_row(observed, 'psilodep1 QIDS_1week (observed)', reference_sd)]
    rows += [spread_row(frame['prediction'],
                        f'graphTRIP zero-shot prediction ({condition})', reference_sd)
             for condition, frame in predictions.items()]

    table = pd.DataFrame(rows)
    out.table('c_prediction_spread', table)
    out.log_df('Spread of the training outcome, the validation outcome and the '
               'zero-shot predictions', table)
    return table


@register('fig4', group='main', subdir='Fig.4')
def fig4_validation(ctx, out):
    results_base_dir = require(output_dir('validation', 'evaluate_graphtrip'))

    # b. Baseline severity of the two cohorts --------------------------------------------
    baseline_tests = baseline_severity_panels(
        out, 'b_baseline_severity', np.random.default_rng(ctx.cfg.seed))
    out.table('b_baseline_severity_tests', baseline_tests)
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
                             'd_reconstruction_corrs')

    # c. Zero-shot prediction ------------------------------------------------------------
    results = zeroshot_results(results_base_dir, PANEL_SUFFIX)
    pvalue = zeroshot_pvalue(out, results, ctx.num_seeds)
    zeroshot_scatter(results, out, 'c_true_vs_pred_harmonised', pvalue)

    prediction_spread_table(out)
