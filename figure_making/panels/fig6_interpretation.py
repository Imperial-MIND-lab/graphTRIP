"""
Fig. 6: Interpreting predictions (regional attributions and GRAIL biomarkers).

Panels:
- b. regional attributions on the brain surface
- c. regional attributions per resting-state network
- d. dominance analysis of regional attribution patterns
- f. GRAIL biomarker alignments across prediction modes

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import pandas as pd

from utils.plotting import CMAP_DEFAULT, COOLWARM, plot_brain_surface, plot_diverging_bars_multi

from figure_making.common import (
    load_biomarker_categories, rsn_attribution_panels, dominance_panels)
from figure_making.paths import posthoc_dir, require
from figure_making.registry import register


ALIGNMENT_MODES = [
    ('graphtrip', 'grail'),
    ('medusa_graphtrip', 'grail_psilocybin'),
    ('medusa_graphtrip', 'grail_escitalopram'),
]


@register('fig6', group='main', subdir='Fig.6')
def fig6_interpretation(ctx, out):

    # b. Regional attributions on the brain surface -------------------------------------
    for model, panel_prefix, brain_subdir in [
            ('graphtrip', '', 'graphtrip_regional_attributions'),
            ('medusa_graphtrip', 'medusa_', 'medusa_graphtrip_regional_attributions')]:

        results_dir = require(posthoc_dir(model, 'regional_attributions'))
        attributions = pd.read_csv(os.path.join(results_dir, 'weighted_mean_attributions.csv'))
        population_mean = attributions.mean(axis=0).values

        plot_brain_surface(population_mean,
                           atlas=ctx.atlas,
                           cmap=CMAP_DEFAULT,
                           save_path=out.fig(f'{panel_prefix}population_mean_regional_attributions',
                                             ext='png'))
        out.brain(f'{brain_subdir}/weighted_mean_regional_attributions', population_mean)

    # c. Regional attributions per resting-state network --------------------------------
    graphtrip_rsn = rsn_attribution_panels(
        posthoc_dir('graphtrip', 'regional_attributions'),
        out, 'mean_rsn_attributions_raincloud')
    out.log_df('graphTRIP: pairwise RSN comparisons', graphtrip_rsn)

    medusa_rsn = rsn_attribution_panels(
        posthoc_dir('medusa_graphtrip', 'regional_attributions'),
        out, 'medusa_mean_rsn_attributions_raincloud')
    out.log_df('Medusa graphTRIP: pairwise RSN comparisons', medusa_rsn)

    # d. Dominance analysis --------------------------------------------------------------
    dominance_panels(posthoc_dir('graphtrip', 'regional_attributions'), out, 'graphtrip')
    dominance_panels(posthoc_dir('medusa_graphtrip', 'regional_attributions'), out, 'medusa')

    # f. GRAIL biomarker alignments across prediction modes ------------------------------
    _, _, sorted_biomarkers = load_biomarker_categories(thresh=0.5)

    dfs = []
    for model, analysis in ALIGNMENT_MODES:
        grail_dir = require(posthoc_dir(model, analysis))
        alignments = pd.read_csv(os.path.join(grail_dir, 'weighted_mean_alignments.csv'))
        dfs.append(alignments[sorted_biomarkers[::-1]])

    plot_diverging_bars_multi(dfs,
                              yline=0,
                              cmap=COOLWARM,
                              vmax=0.15,
                              alpha=0.9,
                              add_scatter=False,
                              scatter_alpha=0.5,
                              scatter_size=20,
                              figsize=(7, 5),
                              save_path=out.fig('mean_alignments_all_modes'),
                              bar_orientation='horizontal',
                              bar_width=0.5,
                              add_colorbar=False)

    out.log(f'Biomarkers shown (majority-significant in >=50% of subjects): '
            f'{len(sorted_biomarkers)}')
    out.log(', '.join(sorted_biomarkers))
