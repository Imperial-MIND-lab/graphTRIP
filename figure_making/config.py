"""
Configuration for figure generation.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
from dataclasses import dataclass

from utils.files import add_project_root


DEFAULT_OUTDIR = 'notebooks/figures'
FORMATS = ['svg', 'png']


@dataclass
class FigureConfig:
    """Settings shared by every figure target."""
    outdir: str = DEFAULT_OUTDIR
    fmt: str = 'svg'
    save: bool = True
    seed: int = 0
    verbose: bool = False

    def __post_init__(self):
        if self.fmt not in FORMATS:
            raise ValueError(f"Invalid format: {self.fmt}. Must be one of {FORMATS}.")
        self.outdir = add_project_root(self.outdir)

    @property
    def brain_plots_dir(self):
        '''Directory for regional values plotted on a brain surface (MATLAB BrainNetViewer).'''
        return os.path.join(self.outdir, 'brain_plots')


def init_matplotlib(cfg: FigureConfig):
    '''
    Configures matplotlib for batch figure generation.

    Must be called before utils.plotting is imported, because that module imports
    pyplot at line 2 and the backend cannot be switched afterwards.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Keep text as text in svg output, so panels stay editable in Illustrator/Inkscape.
    plt.rcParams['svg.fonttype'] = 'none'

    # Derive svg element ids from the seed rather than at random, so that regenerating a
    # panel with unchanged data produces an unchanged file (apart from its date stamp).
    plt.rcParams['svg.hashsalt'] = str(cfg.seed)

    # No plotting function in utils/plotting.py calls plt.close, and most call an
    # unconditional plt.show() which is a no-op under Agg. Figures therefore
    # accumulate until the runner closes them after each target.
    plt.rcParams['figure.max_open_warning'] = 0
