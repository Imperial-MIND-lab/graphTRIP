"""
Trains the nested cross-validation models used to validate the biomarker selection.

One run trains the inner ensemble of one (outer fold, seed) pair. The outer fold is held
out completely. Everything else matches the published graphTRIP pipeline.

--summary reads the runs back and compares the inner fold performance against the published
models. Run it before spending GRAIL compute: if the inner models do not predict at the
reduced training size, nothing downstream is worth computing.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/graphtrip/nested_cv/outer_{f}/weights/seed_{s}/
- outputs/graphtrip/nested_cv/analysis/nested_cv_performance.csv  (--summary)

Author: Hanna M. Tolle
Date: 2026-09-12
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import glob
import copy
import argparse
import numpy as np
import pandas as pd

from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from utils.statsalg import get_fold_performance
from experiments.run_experiment import run


DEFAULT_CONFIG = 'experiments/configs/graphtrip.json'
DEFAULT_OUTPUT_DIR = 'outputs/graphtrip/nested_cv/'
PUBLISHED_WEIGHTS_DIR = 'outputs/graphtrip/weights/'


def weights_dir(output_dir, outer_fold, seed):
    '''The weights of one (outer fold, seed) run.'''
    return os.path.join(output_dir, f'outer_{outer_fold}', 'weights', f'seed_{seed}')


def main(config_file, output_dir, verbose, debug, seed, outer_fold,
         num_outer_folds=7, outer_seed=0, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)

    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    config['outer_fold'] = outer_fold
    config['num_outer_folds'] = num_outer_folds
    config['outer_seed'] = outer_seed
    if debug:
        config['num_epochs'] = 2

    # Train the inner ensemble of this outer fold ------------------------------
    exname = 'train_nested'
    ex_dir = weights_dir(output_dir, outer_fold, seed)

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        run(exname, observer, config_updates)
    else:
        print(f"Nested CV experiment already exists in {ex_dir}.")


def fold_performances(base_dir, pattern):
    '''
    Per-fold performance of every run under base_dir, as one frame.

    pattern globs the run directories and must capture the outer fold and seed in its
    path; runs that have not finished are skipped rather than raising.
    '''
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(base_dir, pattern))):
        if not os.path.exists(os.path.join(run_dir, 'prediction_results.csv')):
            continue
        performance = get_fold_performance(run_dir)
        performance['run'] = os.path.relpath(run_dir, base_dir)
        rows.append(performance)
    if not rows:
        raise FileNotFoundError(f'No finished runs in {os.path.join(base_dir, pattern)}')
    return pd.concat(rows, ignore_index=True)


def describe(performance, name):
    '''One line summarising a set of fold models.'''
    rho = performance['rho']
    return {'models': name, 'n_models': len(rho),
            'rho_median': rho.median(), 'rho_q25': rho.quantile(0.25),
            'rho_q75': rho.quantile(0.75), 'rho_mean': rho.mean(),
            'n_rho_positive': int((rho > 0).sum()),
            'frac_rho_positive': float((rho > 0).mean()),
            'r_median': performance['r'].median()}


def summary(output_dir):
    '''
    Compares the nested inner fold models against the published ones.

    This is the go/no-go gate: the nested models train on fewer patients, and if that
    costs them their predictive performance then the selection they support is not worth
    running GRAIL on.
    '''
    output_dir = add_project_root(output_dir)
    nested = fold_performances(output_dir, os.path.join('outer_*', 'weights', 'seed_*'))
    nested['outer_fold'] = nested['run'].str.extract(r'outer_(\d+)').astype(int)
    nested['seed'] = nested['run'].str.extract(r'seed_(\d+)').astype(int)

    rows = [describe(nested, 'nested (inner folds)')]
    try:
        published = fold_performances(add_project_root(PUBLISHED_WEIGHTS_DIR), 'seed_*')
        rows.append(describe(published, 'published'))
    except FileNotFoundError:
        print(f'No published models in {PUBLISHED_WEIGHTS_DIR} to compare against.')

    comparison = pd.DataFrame(rows)
    print('\nInner fold performance\n' + comparison.to_string(index=False))

    print('\nPer outer fold')
    per_outer = (nested.groupby('outer_fold')
                 .agg(n_models=('rho', 'size'), rho_median=('rho', 'median'),
                      n_rho_positive=('rho', lambda x: int((x > 0).sum())))
                 .reset_index())
    print(per_outer.to_string(index=False))

    # The held-out patients, predicted by the ensemble that never saw them
    outer_metrics = []
    for path in sorted(glob.glob(os.path.join(output_dir, 'outer_*', 'weights', 'seed_*',
                                              'outer_metrics.csv'))):
        outer_metrics.append(pd.read_csv(path))
    if outer_metrics:
        outer_metrics = pd.concat(outer_metrics, ignore_index=True)
        print(f'\nHeld-out patients, ensemble prediction over '
              f'{len(outer_metrics)} (outer fold, seed) runs: '
              f"r = {outer_metrics['r'].mean():+.4f} +/- {outer_metrics['r'].std():.4f} "
              f"(median {outer_metrics['r'].median():+.4f})")
        print('Per-run values are noisy: each is 6 patients.')
    else:
        outer_metrics = pd.DataFrame()

    # Write the tables next to the analysis outputs
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    nested.to_csv(os.path.join(analysis_dir, 'nested_cv_performance.csv'), index=False)
    comparison.to_csv(os.path.join(analysis_dir, 'nested_cv_performance_summary.csv'),
                      index=False)
    if len(outer_metrics):
        outer_metrics.to_csv(os.path.join(analysis_dir, 'nested_cv_outer_metrics.csv'),
                             index=False)
    print(f'\nWrote tables to {analysis_dir}')


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.nested_cv -o 0 -s 0 -v
    python -m scripts.nested_cv --summary
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG,
                        help='Path to the config file')
    parser.add_argument('-out', '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Path to the output directory')
    parser.add_argument('-o', '--outer_fold', type=int, default=0,
                        help='Which outer fold to hold out')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Training seed')
    parser.add_argument('--num_outer_folds', type=int, default=7,
                        help='Number of outer folds')
    parser.add_argument('--outer_seed', type=int, default=0,
                        help='Seed of the outer split; keep fixed across training seeds')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=0, help='Config ID')
    parser.add_argument('--summary', action='store_true',
                        help='Summarise the trained runs instead of training')
    args = parser.parse_args()

    if args.summary:
        summary(args.output_dir)
    else:
        main(args.config, args.output_dir, args.verbose, args.debug, args.seed,
             args.outer_fold, args.num_outer_folds, args.outer_seed, args.config_id)
