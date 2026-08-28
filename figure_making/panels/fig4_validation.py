"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

graphTRIP is applied zero-shot: no weight is updated and no psilodep1 outcome is used.
The baseline severity scores are harmonised onto the psilodep2 training scale first, which
corrects the one cohort difference that can be identified before any outcome is observed.

Panels:
- b. reconstruction performance of the graphTRIP VGAE, not fine-tuned, tested against
     the primary-dataset reconstructions of Fig. 2d
- c. zero-shot prediction performance with harmonised baseline severity scores

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd

from utils.configs import load_ingredient_configs
from utils.helpers import aggregate_prediction_results
from utils.plotting import true_vs_pred_scatter
from utils.statsalg import compare_reconstruction_performance, correlation_permutation_test

from figure_making.common import plot_correlation_boxplot, collect_seed_metrics
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, require
from figure_making.registry import register


# Must match N_PERMUTATIONS in scripts/validation.py
N_PERMUTATIONS = 10000

# Independent fold assignments drawn for the ensemble-matched reconstruction control
N_MATCHED_DRAWS = 10

CONDITIONS = [('no harmonisation', ''), ('harmonised', '_harmonised')]


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
    (Fig. 2d) and the independent validation dataset (Fig. 4b).
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


def zeroshot_permutation_table(results_base_dir):
    '''
    Permutation test of the zero-shot correlation, for each condition.

    The per-subject predictions are first averaged over seeds (the grand mean vote), then
    the outcomes are shuffled and the correlation recomputed. Because the predictions use
    no psilodep1 outcomes and no weights were fitted here, they are unchanged by the
    shuffle, so this null is exact.

    The per-seed columns report how many of the individually re-trained graphTRIP models
    reach significance on their own.
    '''
    rows = []
    for label, suffix in CONDITIONS:
        results = aggregate_prediction_results(results_file=os.path.join(
            results_base_dir, f'initial_prediction_results_mean_vote{suffix}.csv'))
        perm = correlation_permutation_test(
            results['label'].values, results['prediction'].values,
            n_permutations=N_PERMUTATIONS, seed=0, make_plot=False)

        per_seed = collect_seed_metrics(
            [(label, results_base_dir, f'initial_metrics_mean_vote{suffix}.csv')])
        rows.append({'condition': label,
                     'r': perm['observed_r'],
                     'permutation_p': perm['p_value'],
                     'null_mean': perm['null_mean'],
                     'null_sd': perm['null_std'],
                     'n_permutations': N_PERMUTATIONS,
                     'n_seeds': len(per_seed),
                     'seeds_p_below_0.05': int((per_seed['p'] < 0.05).sum()),
                     'mean_seed_r': per_seed['r'].mean(),
                     'sd_seed_r': per_seed['r'].std()})
    return pd.DataFrame(rows)


@register('fig4', group='main', subdir='Fig.4')
def fig4_validation(ctx, out):
    results_base_dir = require(output_dir('validation', 'evaluate_graphtrip'))

    # b. Reconstruction performance on the validation dataset ---------------------------
    psilodep1_config = load_ingredient_configs(os.path.join(results_base_dir, 'seed_0'),
                                               ingredients=['dataset'])
    psilodep1_data = load_dataset(psilodep1_config['dataset'])
    psilodep1_num_subs = len(psilodep1_data)

    # All psilodep1 patients were treated with psilocybin
    psilodep1_conditions = np.ones(psilodep1_num_subs)

    # Every VGAE of every seed and fold reconstructs every patient, then averages, and
    # the result is tested against the primary dataset of Fig. 2d
    psilodep1_x = reconstruction_comparison(ctx, out, psilodep1_data)

    plot_correlation_boxplot(out, psilodep1_x, psilodep1_conditions,
                             'original_vs_reconstructed_corrs')

    # c. Zero-shot prediction performance, harmonised -----------------------------------
    results = aggregate_prediction_results(results_file=os.path.join(
        results_base_dir, 'initial_prediction_results_mean_vote_harmonised.csv'))
    results['Condition'] = 1  # add psilocybin condition for plotting
    true_vs_pred_scatter(results, save_path=out.fig('zeroshot_harmonised_true_vs_pred'),
                         yerr='prediction_sem')

    # Permutation test of that correlation, against the unharmonised baseline
    permutation_results = zeroshot_permutation_table(results_base_dir)
    out.table('zeroshot_permutation_test', permutation_results)
    out.log_df('Zero-shot permutation test', permutation_results)
