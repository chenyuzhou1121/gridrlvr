"""Build verl Parquet prompts by resetting external ANM or RestoreBench scenarios."""

from __future__ import annotations
import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional
DEFAULT_SYSTEM_PROMPT = (
    'You are an Active Network Management operator restoring a distribution grid.\n'
    'At each turn, inspect the observation and emit exactly one available function call. '
    'The verifier executes every proposed action on a shadow copy. '
    'A FAIL or ERROR action is not applied; use its structured feedback to revise the next call. '
    'Continue until the environment reports recovery. Never claim recovery from text alone.'
)

def _scenario_records(payload: Any, split: Optional[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif not isinstance(payload, Mapping):
        raise TypeError('scenario input must be a JSON list or object')
    elif isinstance(payload.get('scenarios'), list):
        records = payload['scenarios']
    elif split and isinstance(payload.get(split), list) and all((isinstance(x, Mapping) for x in payload[split])):
        records = payload[split]
    else:
        raise ValueError('input must contain full scenario records, not identifiers alone')
    result = [dict(record) for record in records if isinstance(record, Mapping)]
    if split:
        result = [record for record in result if str(record.get('split', split)) == split]
    return result

def load_scenarios(path: str | Path, split: Optional[str]=None) -> list[dict[str, Any]]:
    """Read JSON record lists or scenarios/split containers, optionally filtered."""
    with Path(path).open(encoding='utf-8') as handle:
        return _scenario_records(json.load(handle), split)

def _make_env(factory: Callable[..., Any], scenario_file: str | Path | None, max_steps: int) -> Any:
    kwargs: dict[str, Any] = {'max_steps': max_steps, 'gating_policy': 'progress_mag'}
    if scenario_file is not None:
        kwargs['scenario_file'] = str(scenario_file)
    try:
        return factory(**kwargs)
    except TypeError:
        return factory()

def make_row(scenario: Mapping[str, Any], index: int, *, env_factory: Callable[..., Any], scenario_file: str | Path | None=None, max_steps: int=8, system_prompt: str=DEFAULT_SYSTEM_PROMPT, default_split: str='train') -> dict[str, Any]:
    """Reset one scenario and return a verl row with prompt and simulator metadata.

    seed or rep_seed selects the replicate. The default prompt is replaced by
    env.system_prompt when available. The environment is closed after reset.
    """
    scenario_id = str(scenario.get('scenario_id', scenario.get('id', ''))).strip()
    if not scenario_id:
        raise ValueError(f'scenario at index {index} has no id/scenario_id')
    seed = int(scenario.get('rep_seed', scenario.get('seed', 0)) or 0)
    source_seed = scenario.get('source_seed')
    env = _make_env(env_factory, scenario_file, max_steps)
    try:
        observation = env.reset(scenario_id=scenario_id, seed=seed)
        current_scenario = getattr(env, 'current_scenario', None)
        if source_seed is None and current_scenario is not None:
            source_seed = getattr(current_scenario, 'source_seed', None)
        resolved_system_prompt = system_prompt
        if system_prompt == DEFAULT_SYSTEM_PROMPT:
            resolved_system_prompt = str(getattr(env, 'system_prompt', system_prompt))
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            close()
    split = str(scenario.get('split', default_split))
    difficulty = scenario.get('difficulty', 'unknown')
    return {
        'data_source': 'gridrlvr/anm',
        'agent_name': 'anm_agent_loop',
        'prompt': [
            {'role': 'system', 'content': resolved_system_prompt},
            {'role': 'user', 'content': str(observation)},
        ],
        'ability': 'power_grid_restoration',
        'reward_model': {
            'style': 'rule',
            'ground_truth': json.dumps(
                {'scenario_id': scenario_id, 'criterion': 'simulator_recovered'},
                ensure_ascii=False,
                separators=(',', ':'),
            ),
        },
        'extra_info': {
            'scenario_id': scenario_id,
            'difficulty': difficulty,
            'split': split,
            'seed': seed,
            'source_seed': source_seed,
            'index': index,
            'initial_state_group': f'{scenario_id}:{seed}',
        },
    }

def build_rows(scenarios: Sequence[Mapping[str, Any]], *, env_factory: Callable[..., Any], scenario_file: str | Path | None=None, max_steps: int=8, system_prompt: str=DEFAULT_SYSTEM_PROMPT, default_split: str='train') -> list[dict[str, Any]]:
    """Return one make_row result per scenario, preserving input order."""
    return [make_row(scenario, index, env_factory=env_factory, scenario_file=scenario_file, max_steps=max_steps, system_prompt=system_prompt, default_split=default_split) for (index, scenario) in enumerate(scenarios)]

def write_parquet(rows: Sequence[Mapping[str, Any]], output: str | Path) -> None:
    """Write row mappings as a PyArrow Parquet table, creating parent directories."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('build_dataset requires pyarrow in the verl environment') from exc
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, output_path)

def _default_env_factory(**kwargs: Any) -> Any:
    from gridrlvr.env import ANMEnv
    if 'scenario_file' in kwargs:
        kwargs['scenarios_path'] = kwargs.pop('scenario_file')
    return ANMEnv(**kwargs)

def parse_args() -> argparse.Namespace:
    """Parse scenario input, environment, prompt, and Parquet output options."""
    parser = argparse.ArgumentParser(description='Build verl parquet from external scenario records')
    parser.add_argument('--input', required=True, help='External scenario JSON containing full records')
    parser.add_argument('--environment', choices=('anm', 'restorebench'), default='anm', help='Rollout environment')
    parser.add_argument('--env-url', default='http://127.0.0.1:8737', help='RestoreBench environment service URL')
    parser.add_argument('--corpus', help='Default RestoreBench corpus')
    parser.add_argument('--timeout', type=float, default=600.0, help='Environment HTTP timeout in seconds')
    parser.add_argument('--output', required=True, help='Output parquet path')
    parser.add_argument('--scenario-file', help='Catalog path passed to ANMEnv')
    parser.add_argument('--split', choices=('train', 'held_out', 'heldout'))
    parser.add_argument('--max-steps', type=int, default=8, help='Tool-call budget')
    parser.add_argument('--system-prompt-file', help='UTF-8 text replacing the environment system prompt')
    return parser.parse_args()

def main() -> None:
    """Build prompts for the selected environment and print output path/row count."""
    args = parse_args()
    scenarios = load_scenarios(args.input, split=args.split)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding='utf-8')
    if args.environment == 'anm':
        rows = build_rows(scenarios, env_factory=_default_env_factory, scenario_file=args.scenario_file or args.input, max_steps=args.max_steps, system_prompt=system_prompt, default_split=args.split or 'train')
    else:
        from gridrlvr.env.remote import RestoreBenchRemoteEnv
        rows = []
        for index, scenario in enumerate(scenarios):
            corpus = scenario.get('corpus') or args.corpus
            if not corpus:
                raise SystemExit('RestoreBench requires --corpus or a corpus field per record')
            def factory(**kwargs: Any) -> Any:
                return RestoreBenchRemoteEnv(base_url=args.env_url, corpus=corpus, timeout=args.timeout, terminate_on_propose=False)
            row = make_row(scenario, index, env_factory=factory, max_steps=args.max_steps, system_prompt=system_prompt, default_split=args.split or 'train')
            row['data_source'] = 'gridrlvr/restorebench'
            row['agent_name'] = 'restorebench_agent_loop'
            row['extra_info']['corpus'] = corpus
            row['extra_info']['initial_state_group'] = f"{corpus}:{row['extra_info']['initial_state_group']}"
            rows.append(row)
    write_parquet(rows, args.output)
    print(json.dumps({'output': args.output, 'rows': len(rows)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
