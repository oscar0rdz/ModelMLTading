#!/usr/bin/env python3
"""
Cleanup previously generated outputs so repeated runs don't accumulate old results.

By default this script runs in dry-run mode and prints what would be removed.
Pass --yes to actually delete files.

It deletes the following by default:
- ML/out/wfa_* directories (walk-forward outputs)
- reports/*
- docs/figs/*
- backtest.log

It intentionally does NOT touch ML/results (trained models) or source files.
"""
import argparse
from pathlib import Path
import shutil
import sys


def find_targets():
    cwd = Path(__file__).resolve().parents[1]
    targets = []
    # ML/out/wfa_* directories
    for p in (cwd / 'ML' / 'out').glob('wfa_*'):
        targets.append(p)
    # reports/
    for p in (cwd / 'reports').glob('*'):
        targets.append(p)
    # docs/figs/
    for p in (cwd / 'docs' / 'figs').glob('*'):
        targets.append(p)
    # backtest.log
    if (cwd / 'backtest.log').exists():
        targets.append(cwd / 'backtest.log')
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true', help='Actually delete files')
    parser.add_argument('--what', default=None, help='Optional glob to target a subset')
    args = parser.parse_args()

    targets = find_targets()
    if args.what:
        from fnmatch import fnmatch
        targets = [p for p in targets if fnmatch(str(p), args.what)]

    if not targets:
        print('Nothing to remove.')
        return

    print('Found {} items:'.format(len(targets)))
    for p in targets:
        t = 'dir' if p.is_dir() else 'file'
        print(f'- {t}: {p}')

    if not args.yes:
        print('\nDry-run: no files were deleted. Re-run with --yes to delete these items.')
        return

    # perform deletion
    removed = []
    for p in targets:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(p)
        except Exception as e:
            print(f'ERROR removing {p}: {e}', file=sys.stderr)

    print('\nRemoved {} items:'.format(len(removed)))
    for p in removed:
        print(f'- {p}')


if __name__ == '__main__':
    main()
