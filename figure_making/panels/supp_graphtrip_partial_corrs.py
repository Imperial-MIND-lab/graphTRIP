"""
Supplementary: partial correlation of the graphTRIP predictions with the outcome,
controlling for treatment condition, baseline QIDS and their interaction.

Author: Hanna M. Tolle
Date: 2026-08-18
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_prediction_results

from figure_making.common import attach_annotations, partial_correlation_panels
from figure_making.paths import require
from figure_making.registry import register


@register('graphtrip_partial_corrs', group='supp',
          subdir='SUPPLEMENTARY/graphtrip_partial_corrs')
def graphtrip_partial_corrs(ctx, out):
    '''Partial correlation of graphTRIP's predictions.'''
    results_file = os.path.join(ctx.weights_base_dir, 'prediction_results.csv')
    require(os.path.dirname(results_file))
    results = attach_annotations(aggregate_prediction_results(results_file=results_file))

    summary = partial_correlation_panels(results, out, 'graphtrip_partial_corr')
    out.log_df('graphTRIP partial correlations', summary)
    out.table('graphtrip_partial_correlations', summary)
