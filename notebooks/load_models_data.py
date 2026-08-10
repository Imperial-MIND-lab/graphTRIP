"""
Functions for loading models and datasets from experiment folders.

The implementation moved to figure_making/loaders.py; this module re-exports it so
that the notebooks can keep importing it as `from load_models_data import ...`.

Author: Hanna Tolle
Date: 2025-01-12
License: BSD-3-Clause
"""

import sys
sys.path.append('../')

from figure_making.loaders import (
    init_vgae,
    load_vgaes,
    init_mlp,
    load_mlps,
    load_dataset,
)

__all__ = ['init_vgae', 'load_vgaes', 'init_mlp', 'load_mlps', 'load_dataset']
