"""
Supplementary: partial correlations of graphTRIP and the clinical-only MLP benchmark.

Author: Hanna M. Tolle
Date: 2026-08-18
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_prediction_results

from figure_making.common import attach_annotations, partial_correlation_panels
from figure_making.panels.fig2_performance import BENCHMARK_DIR
from figure_making.paths import output_dir, require
from figure_making.registry import register


@register('graphtrip_partial_corrs', group='supp',
          subdir='SUPPLEMENTARY/graphtrip_partial_corrs')
def graphtrip_partial_corrs(ctx, out):
    '''Partial correlations of graphTRIP and of the clinical-only MLP benchmark.'''
    # The scatters themselves are Fig.2a, so only the residual panels are drawn here.
    results_file = os.path.join(ctx.weights_base_dir, 'prediction_results.csv')
    require(os.path.dirname(results_file))
    results = attach_annotations(aggregate_prediction_results(results_file=results_file))

    benchmark_file = os.path.join(output_dir(*BENCHMARK_DIR), 'prediction_results.csv')
    require(os.path.dirname(benchmark_file))
    benchmark_results = attach_annotations(
        aggregate_prediction_results(results_file=benchmark_file))

    out.log('=== graphTRIP ===')
    summary = partial_correlation_panels(results, out, 'graphtrip_partial_corr')
    out.log_df('graphTRIP partial correlations', summary)
    out.table('graphtrip_partial_correlations', summary)

    out.log('=== Clinical-only MLP benchmark ===')
    benchmark_summary = partial_correlation_panels(
        benchmark_results, out, 'control_mlp_partial_corr')
    out.log_df('Clinical-only MLP partial correlations', benchmark_summary)
    out.table('control_mlp_partial_correlations', benchmark_summary)
