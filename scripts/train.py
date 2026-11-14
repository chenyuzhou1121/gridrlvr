#!/usr/bin/env python3
"""Translate repository YAML and CLI overrides into a verl training command."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def flatten(config: dict[str, Any], prefix: str = '') -> list[str]:
    """Convert nested config leaves to Hydra force-add overrides under prefix."""
    result = []
    for key, value in config.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict) and value:
            result.extend(flatten(value, name))
        elif isinstance(value, dict):
            result.append(f'++{name}={{}}')
        else:
            result.append(f'++{name}={json.dumps(value)}')
    return result


def main(argv: list[str] | None = None) -> int:
    """Resolve launcher options and return the training subprocess exit status.

    Print-only mode returns zero without launching verl. Explicit CLI options
    override YAML, and positional Hydra overrides take final precedence.
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Launch verl v0.9.0 training from portable configuration')
    parser.add_argument('--config', type=Path, default=root / 'configs/train.yaml', help='Training configuration YAML')
    parser.add_argument('--environment', choices=('anm', 'restorebench'), default='anm', help='Rollout environment')
    parser.add_argument('--model-path', help='Policy model directory')
    parser.add_argument('--train-data', help='Training parquet path')
    parser.add_argument('--val-data', help='Validation parquet path')
    parser.add_argument('--scenario-file', help='External ANM scenario catalog')
    parser.add_argument('--env-url', help='RestoreBench environment service URL')
    parser.add_argument('--reward-dimensions', action='store_true', help='Use GDPO per-constraint rewards for ANM; requires reward patch')
    parser.add_argument('--gpus-per-node', type=int, help='GPU count per node')
    parser.add_argument('--checkpoint-dir', type=Path, help='Checkpoint destination')
    parser.add_argument('--log-file', type=Path, default=Path('training.log'), help='Launcher log destination')
    parser.add_argument('--print-command', action='store_true', help='Print resolved command without starting training')
    parser.add_argument('overrides', nargs='*', help='Additional verl Hydra overrides')
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise SystemExit('Training configuration must be a mapping')
    backend = args.environment
    if args.reward_dimensions and backend != 'anm':
        raise SystemExit('Per-constraint rewards require the ANM environment')
    overrides = flatten(config)
    loop_path = root / 'configs' / ('anm_loop.yaml' if backend == 'anm' else 'restorebench_loop.yaml')
    overrides += [f'++actor_rollout_ref.rollout.agent.agent_loop_config_path={json.dumps(str(loop_path))}']
    overrides += [f'++actor_rollout_ref.rollout.agent.default_agent_loop={"anm_agent_loop" if backend == "anm" else "restorebench_agent_loop"}']
    reward_path = root / 'gridrlvr/verl_adapter' / 'reward_fn.py'
    reward_name = 'compute_score' if backend == 'anm' else 'compute_score_restorebench'
    overrides += [f'++reward.custom_reward_function.path={json.dumps(str(reward_path))}', f'++reward.custom_reward_function.name={reward_name}']
    for key, value in [('actor_rollout_ref.model.path', args.model_path), ('data.train_files', args.train_data), ('data.val_files', args.val_data), ('trainer.n_gpus_per_node', args.gpus_per_node), ('trainer.default_local_dir', str(args.checkpoint_dir) if args.checkpoint_dir else None)]:
        if value is not None:
            overrides.append(f'++{key}={json.dumps(value)}')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    if args.scenario_file:
        env['GRIDRLVR_SCENARIO_FILE'] = str(Path(args.scenario_file).expanduser().resolve())
    if args.env_url:
        env['GRIDRLVR_RESTOREBENCH_ENV_URL'] = args.env_url
    if args.reward_dimensions:
        env['GRIDRLVR_ANM_LOOP_TARGET'] = 'gridrlvr.verl_adapter.reward_gdpo.GDPOANMAgentLoop'
        overrides += ['++algorithm.adv_estimator=gdpo', '++algorithm.gdpo_reward_keys=[voltage_reward,branch_loading_reward,storage_soc_reward,recovered_reward]', '++algorithm.gdpo_reward_weights=[1.0,1.0,1.0,1.0]']
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', *overrides, *args.overrides]
    if args.print_command:
        import shlex
        print(shlex.join(command))
        return 0
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)])
    logging.info('Starting verl training with environment=%s', backend)
    result = subprocess.run(command, env=env)
    logging.info('Training process exited with status=%s', result.returncode)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
