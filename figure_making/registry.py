"""
Registry of figure targets and the runner that executes them.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import traceback
from collections import OrderedDict

import matplotlib.pyplot as plt

from figure_making.io import FigureOutput
from figure_making.paths import MissingInput


TARGETS = OrderedDict()
GROUPS = ['main', 'supp']


class Target:
    def __init__(self, name, func, group, subdir):
        self.name = name
        self.func = func
        self.group = group
        self.subdir = subdir


def register(name, group, subdir):
    '''
    Registers a figure target.

    Parameters:
    ----------
        name (str): Target name used on the command line.
        group (str): 'main' or 'supp'.
        subdir (str): Output directory relative to the figure root, e.g. 'Fig.2'.
    '''
    if group not in GROUPS:
        raise ValueError(f"Invalid group: {group}. Must be one of {GROUPS}.")

    def decorator(func):
        if name in TARGETS:
            raise ValueError(f"Duplicate figure target: {name}")
        TARGETS[name] = Target(name, func, group, subdir)
        return func
    return decorator


def resolve(names):
    '''
    Expands the requested target names into a list of Targets.

    An empty list means all targets; 'main' and 'supp' expand to their groups.
    '''
    if not names:
        return list(TARGETS.values())

    resolved = []
    for name in names:
        if name in GROUPS:
            resolved.extend(t for t in TARGETS.values() if t.group == name)
        elif name in TARGETS:
            resolved.append(TARGETS[name])
        else:
            raise ValueError(
                f"Unknown figure target: {name}.\n"
                f"Available targets: {', '.join(TARGETS)}\n"
                f"Available groups: {', '.join(GROUPS)}")

    # Deduplicate while preserving registration order
    return [t for t in TARGETS.values() if t in resolved]


def list_targets():
    '''Returns a printable listing of all registered targets.'''
    lines = []
    for group in GROUPS:
        lines.append(f'{group.upper()}:')
        for target in TARGETS.values():
            if target.group == group:
                lines.append(f'    {target.name:<50} -> {target.subdir}/')
        lines.append('')
    return '\n'.join(lines)


def run_targets(targets, ctx):
    '''
    Runs each target, skipping those whose input results are missing.

    Returns (completed, skipped), where skipped is a list of (name, reason).
    '''
    completed, skipped = [], []

    for target in targets:
        out = FigureOutput(ctx.cfg, target.subdir)
        try:
            target.func(ctx, out)
        except MissingInput as e:
            print(f'[skip] {target.name}  -- missing {e}', flush=True)
            skipped.append((target.name, f'missing {e}'))
            continue
        except FileNotFoundError as e:
            print(f'[skip] {target.name}  -- missing {e.filename or e}', flush=True)
            skipped.append((target.name, f'missing {e.filename or e}'))
            continue
        except Exception as e:
            print(f'[FAIL] {target.name}  -- {type(e).__name__}: {e}', flush=True)
            if ctx.cfg.verbose:
                traceback.print_exc()
            skipped.append((target.name, f'{type(e).__name__}: {e}'))
            continue
        finally:
            out.flush()
            # utils.plotting never closes its figures; release them between targets.
            plt.close('all')

        print(f'[ok]   {target.name}  -- {out.n_figs} panels, {out.n_tables} tables', flush=True)
        completed.append(target.name)

    return completed, skipped


def print_summary(targets, completed, skipped):
    '''Prints the end-of-run summary of completed and skipped targets.'''
    print()
    print('=== SUMMARY ===')
    print(f'{len(completed)}/{len(targets)} targets completed')
    if skipped:
        print('skipped: ' + ', '.join(f'{name} ({reason})' for name, reason in skipped))
