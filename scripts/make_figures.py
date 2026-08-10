"""
This scripts generates the paper figure panels from the experiment outputs.

Replaces notebooks/figures.ipynb. Each figure target writes its panels, tables and a
stats.txt report into the directory the paper figure is assembled from.

Dependencies:
- outputs/graphtrip/
- outputs/medusa_graphtrip/
- outputs/ablation/
- outputs/medusa_ablation/
- outputs/validation/
- outputs/graphtrip_bdi/
- outputs/biomarker_categories/

Outputs:
- notebooks/figures/Fig.2/ ... Fig.6/
- notebooks/figures/SUPPLEMENTARY/
- notebooks/figures/brain_plots/

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
import os
# Ensure project root is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from figure_making.config import FigureConfig, FORMATS, DEFAULT_OUTDIR, init_matplotlib


def main(targets, outdir, fmt, no_save, seed, verbose):
    cfg = FigureConfig(outdir=outdir, fmt=fmt, save=not no_save, seed=seed, verbose=verbose)
    init_matplotlib(cfg)

    # Imported after the backend is set, because utils.plotting imports pyplot
    from figure_making.context import FigureContext
    from figure_making.registry import resolve, run_targets, print_summary
    import figure_making.panels  # noqa: F401  (registers every target)

    selected = resolve(targets)
    print(f'Generating {len(selected)} figure targets as .{cfg.fmt} in {cfg.outdir}')
    if not cfg.save:
        print('Running with --no-save: no panels or tables will be written.')
    print()

    ctx = FigureContext(cfg)
    completed, skipped = run_targets(selected, ctx)
    print_summary(selected, completed, skipped)


def print_targets():
    import figure_making.panels  # noqa: F401
    from figure_making.registry import list_targets
    print(list_targets())


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.make_figures
    python -m scripts.make_figures fig2 fig6 --fmt png
    python -m scripts.make_figures main --fmt svg -o notebooks/figures_2026-08-10 -v
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', type=str, nargs='*',
                        help='Figure targets to generate, or the groups "main"/"supp". '
                             'Generates all targets if omitted.')
    parser.add_argument('-o', '--outdir', type=str, default=DEFAULT_OUTDIR,
                        help='Path to the figure output directory')
    parser.add_argument('--fmt', type=str, default='svg', choices=FORMATS,
                        help='Panel file format. Brain surface panels are always png.')
    parser.add_argument('--no_save', '--no-save', action='store_true', dest='no_save',
                        help='Run the analyses without writing panels or tables')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--list', action='store_true', help='List figure targets and exit')
    args = parser.parse_args()

    if args.list:
        print_targets()
        sys.exit(0)

    # Run the main function
    main(args.targets, args.outdir, args.fmt, args.no_save, args.seed, args.verbose)
