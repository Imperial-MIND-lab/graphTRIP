"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

graphTRIP is applied zero-shot: no weight is updated and no psilodep1 outcome is used.
The baseline severity scores are harmonised onto the psilodep2 training scale first, which
corrects the one cohort difference that can be identified before any outcome is observed.

Panels:
- b. reconstruction performance of the graphTRIP VGAE, not fine-tuned
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

CONDITIONS = [('no harmonisation', ''), ('harmonised', '_harmonised')]


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

    # Every VGAE of every seed and fold reconstructs every patient, then averages
    _, psilodep1_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, None)

    # Compare reconstruction performance on the primary versus validation dataset
    _, primary_x = ctx.core_reconstructions
    test_results = compare_reconstruction_performance(primary_x['metrics'],
                                                      psilodep1_x['metrics'])
    out.table('primary_vs_validation_corr', test_results['corr'])
    out.table('primary_vs_validation_mae', test_results['mae'])
    out.log_df('Primary vs validation reconstruction (correlation)', test_results['corr'])
    out.log_df('Primary vs validation reconstruction (MAE)', test_results['mae'])

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
