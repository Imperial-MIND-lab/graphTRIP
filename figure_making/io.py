"""
Output handling for figure panels: format-aware save paths, tables and stats logs.

Replaces the notebook idiom
    save_path = None if not save_figs else os.path.join(fig_subdir, 'name.svg')
with
    save_path = out.fig('name')

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd


class FigureOutput:
    """
    Writes the panels, tables and statistics of a single figure target.

    Paths are only created when saving is enabled; with save=False every writer
    is a no-op and fig() returns None, which every utils.plotting function
    interprets as 'do not save'.
    """

    def __init__(self, cfg, subdir):
        self.cfg = cfg
        self.subdir = subdir
        self.dir = os.path.join(cfg.outdir, subdir) if subdir else cfg.outdir
        self._log_lines = []
        self.n_figs = 0
        self.n_tables = 0

    def _ensure_dir(self):
        '''
        Creates the output directory on first write, so that a target skipped for
        missing inputs does not leave an empty directory behind.
        '''
        os.makedirs(self.dir, exist_ok=True)

    # Figures ---------------------------------------------------------------------------

    def fig(self, name, ext=None):
        '''
        Returns the save path for a panel, or None when saving is disabled.

        An extension is always appended: true_vs_pred_scatter and regression_scatter2
        derive the file format from save_path.split('.')[-1], and the figure
        directories contain dots ('Fig.2'), so an extension-less path would be
        interpreted as a format.

        Parameters:
        ----------
            name (str): Filename without extension.
            ext (str): Overrides the configured format. Used for the brain surface
                       panels, which are raster regardless of --fmt.
        '''
        if not self.cfg.save:
            return None
        self._ensure_dir()
        self.n_figs += 1
        return os.path.join(self.dir, f'{name}.{ext or self.cfg.fmt}')

    # Tables and text -------------------------------------------------------------------

    def table(self, name, df, index=False):
        '''Writes a dataframe as <name>.csv into this target's directory.'''
        if not self.cfg.save:
            return None
        self._ensure_dir()
        self.n_tables += 1
        path = os.path.join(self.dir, f'{name}.csv')
        df.to_csv(path, index=index)
        return path

    def text(self, name, content):
        '''Writes text as <name>.txt into this target's directory.'''
        if not self.cfg.save:
            return None
        self._ensure_dir()
        path = os.path.join(self.dir, f'{name}.txt')
        with open(path, 'w') as f:
            f.write(content if content.endswith('\n') else content + '\n')
        return path

    def brain(self, relpath, data):
        '''
        Writes regional values for surface plotting with MATLAB BrainNetViewer into
        <outdir>/brain_plots/<relpath>.csv.

        DataFrames are written with a header (as in the reconstruction panels);
        arrays are written with np.savetxt (as in the regional attribution panels).
        '''
        if not self.cfg.save:
            return None
        path = os.path.join(self.cfg.brain_plots_dir, f'{relpath}.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            np.savetxt(path, np.asarray(data))
        return path

    # Statistics ------------------------------------------------------------------------

    def log(self, msg=''):
        '''
        Records a line of the target's statistics report.

        Replaces print() at the notebook call sites, so numbers that previously only
        existed in cell output are persisted to <dir>/stats.txt.
        '''
        msg = str(msg)
        self._log_lines.append(msg)
        if self.cfg.verbose:
            print(msg)

    def log_df(self, title, df, index=False):
        '''Records a dataframe in the statistics report, formatted as a table.'''
        self.log(title)
        self.log(df.to_string(index=index))
        self.log()

    def flush(self):
        '''Writes the accumulated statistics report to <dir>/stats.txt.'''
        if not self._log_lines or not self.cfg.save:
            return None
        self._ensure_dir()
        path = os.path.join(self.dir, 'stats.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(self._log_lines) + '\n')
        return path
