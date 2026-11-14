#!/usr/bin/env python3
"""Check or apply the bundled reward and FSDP patches to a verl source tree."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Parse patch selections, skip applied patches, and return zero on success."""
    parser = argparse.ArgumentParser(description='Apply optional reward and FSDP compatibility patches to verl v0.9.0')
    parser.add_argument('verl_source', type=Path, help='Source root containing the verl package')
    parser.add_argument('--patches', nargs='+', choices=('rewards', 'offload'), default=['rewards', 'offload'], help='Patch groups to apply')
    parser.add_argument('--check', action='store_true', help='Check applicability without changing files')
    args = parser.parse_args(argv)
    target = args.verl_source.expanduser().resolve()
    if not (target / 'verl').is_dir():
        raise SystemExit('Source root must contain a verl directory')
    patch_root = Path(__file__).resolve().parents[1] / 'patches'
    names = {'rewards': 'verl_reward_dimensions.diff', 'offload': 'verl_fsdp_offload.diff'}
    pending = []
    for group in dict.fromkeys(args.patches):
        patch = patch_root / names[group]
        command = ['git', '-C', str(target), 'apply']
        # A successful reverse check identifies a patch already in the tree.
        reverse = subprocess.run([*command, '--reverse', '--check', str(patch)], capture_output=True, text=True)
        if reverse.returncode == 0:
            print(f'{group}: already applied')
            continue
        forward = subprocess.run([*command, '--check', str(patch)], capture_output=True, text=True)
        if forward.returncode:
            raise SystemExit(f'{group}: patch does not apply to this source tree\n{forward.stderr}')
        pending.append(patch)
    if pending and not args.check:
        subprocess.run(['git', '-C', str(target), 'apply', *map(str, pending)], check=True)
    for patch in pending:
        print(f'{patch.name}: {"applicable" if args.check else "applied"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
