#!/usr/bin/env python3
"""Sample restoration trajectories, replay candidates, and write JSON results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for endpoint, scenario, budget, and output options."""
    parser = argparse.ArgumentParser(description='Generate recovery candidates and select by independent simulator replay')
    parser.add_argument('--environment', choices=('anm', 'restorebench'), default='anm', help='Simulator environment')
    parser.add_argument('--base-url', required=True, help='Chat endpoint, such as https://api.example.com/v1')
    parser.add_argument('--model', required=True, help='Model identifier advertised by the endpoint')
    parser.add_argument('--scenarios', type=Path, help='External JSON or JSONL scenario catalog')
    parser.add_argument('--split', help='Optional catalog split')
    parser.add_argument('--scenario-id', action='append', help='Scenario identifier; repeat to select several')
    parser.add_argument('--corpus', help='RestoreBench corpus registered with the environment service')
    parser.add_argument('--env-url', default='http://127.0.0.1:8737', help='RestoreBench environment service URL')
    parser.add_argument('--candidates', type=int, default=3, help='Independent policy samples per initial state')
    parser.add_argument('--max-steps', type=int, default=8, help='Tool interaction steps per candidate')
    parser.add_argument('--max-fail-retries', type=int, default=2, help='Revisions after a rejected action')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum generated tokens per request')
    parser.add_argument('--temperature', type=float, default=0.7, help='Policy sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.8, help='Nucleus sampling probability')
    parser.add_argument('--seed', type=int, default=0, help='Policy sampling seed')
    parser.add_argument('--env-seed', type=int, default=0, help='Shared initial-state replicate seed')
    parser.add_argument('--timeout', type=float, default=600.0, help='HTTP request timeout in seconds')
    parser.add_argument('--output-dir', required=True, type=Path, help='Destination for generated records')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate selected scenarios and return zero after writing the summary."""
    args = build_parser().parse_args(argv)
    if min(args.candidates, args.max_steps, args.max_tokens) < 1 or args.max_fail_retries < 0:
        raise SystemExit('Budgets must be positive and retries nonnegative')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridrlvr.agent.harness import load_scenarios
    from gridrlvr.agent.react import OpenAICompatibleClient, ReActConfig, ReActRunner, SamplingConfig
    from gridrlvr.agent.selection import select_candidate

    records = load_scenarios(args.scenarios, args.split) if args.scenarios else []
    if args.scenario_id:
        by_id = {item['scenario_id']: item for item in records}
        records = [by_id.get(sid, {'scenario_id': sid}) for sid in args.scenario_id]
    if not records:
        raise SystemExit('Provide --scenarios or --scenario-id')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / 'evaluation.log')])
    client = OpenAICompatibleClient(args.base_url, args.model, timeout_s=args.timeout)
    results = []
    for index, scenario in enumerate(records):
        scenario_id = scenario['scenario_id']
        if args.environment == 'anm':
            from gridrlvr.env import ANMEnv
            # Each rejected proposal consumes one environment step.
            env_factory = lambda: ANMEnv(
                scenarios_path=args.scenarios,
                max_steps=args.max_steps * (args.max_fail_retries + 1),
            )
        else:
            from gridrlvr.env.remote import RestoreBenchRemoteEnv
            corpus = scenario.get('corpus') or args.corpus
            if not corpus:
                raise SystemExit('RestoreBench requires --corpus or a corpus field per scenario')
            env_factory = lambda: RestoreBenchRemoteEnv(base_url=args.env_url, corpus=corpus, timeout=args.timeout, terminate_on_propose=False)
        sampling = SamplingConfig(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
        candidates = []
        for sample in range(args.candidates):
            config = ReActConfig(max_steps=args.max_steps, max_fail_retries=args.max_fail_retries, sampling=replace(sampling, seed=args.seed + sample))
            candidates.append(ReActRunner(env_factory, client, config).run_episode(scenario_id, seed=args.env_seed))
        selection = select_candidate(candidates, env_factory, scenario_id=scenario_id, seed=args.env_seed)
        result = {'scenario_id': scenario_id, 'corpus': scenario.get('corpus', args.corpus), 'candidates': candidates, **selection}
        target = args.output_dir / f'episode_{index:06d}.json'
        target.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
        results.append({'scenario_id': scenario_id, 'recovered': selection['selected']['recovered'], 'selected_index': selection['selected_index'], 'record': target.name})
        logging.info('scenario=%s recovered=%s selected=%s', scenario_id, selection['selected']['recovered'], selection['selected_index'])
    summary = {'episodes': len(results), 'recovered': sum(row['recovered'] for row in results), 'candidates_per_scenario': args.candidates, 'results': results}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
