"""
Path builders for experiment outputs consumed by the figure panels.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import glob
import os

from utils.files import project_root


class MissingInput(Exception):
    '''Raised when a results file or directory a panel depends on does not exist.'''
    pass


def output_dir(*parts):
    '''Returns a path inside outputs/, e.g. output_dir("graphtrip", "weights").'''
    return os.path.join(project_root(), 'outputs', *parts)


def require(path):
    '''Returns path, or raises MissingInput if it does not exist.'''
    if not os.path.exists(path):
        raise MissingInput(path)
    return path


def require_results(*parts):
    '''Returns a path inside outputs/, raising MissingInput if it does not exist.'''
    return require(output_dir(*parts))


# Frequently used results locations ------------------------------------------------------

def graphtrip_weights_dir():
    return output_dir('graphtrip', 'weights')


def medusa_weights_dir():
    return output_dir('medusa_graphtrip', 'weights')


def posthoc_dir(model, analysis):
    '''Post-hoc analysis directory, e.g. posthoc_dir("graphtrip", "grail").'''
    return output_dir(model, analysis, 'posthoc_analysis')


def biomarker_categories_file():
    return output_dir('biomarker_categories', 'biomarker_categories.csv')


def perm_dirs(base):
    '''The perm_* subdirectories of a permutation-null tree, in permutation order.'''
    dirs = glob.glob(os.path.join(base, 'perm_*'))
    return sorted(dirs, key=lambda d: int(os.path.basename(d).split('_')[-1]))
