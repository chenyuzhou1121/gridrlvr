"""Replay sampled trajectories and select one by simulator outcomes."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

_DIAGNOSTIC_TOOLS = {'get_grid_topology', 'get_action_applicability', 'run_ac_pf', 'rank_candidate_maneuvers'}


def replay_candidate(candidate: Mapping[str, Any], env_factory: Callable[[], Any], *, scenario_id: str, seed: int = 0) -> dict[str, Any]:
    """Replay candidate steps in a fresh environment and return verification data.

    Only steps containing an action mapping are replayed. Diagnostic calls do
    not count as accepted or rejected actions; replay errors mark recovery false.
    """
    env = env_factory()
    steps = []
    accepted_actions = 0
    rejected_actions = 0
    violation_count = 0
    process_reward = 0.0
    error = None
    recovered = False
    try:
        env.reset(scenario_id=scenario_id, seed=seed)
        recovered = bool(getattr(env, 'recovered', False))
        for step in candidate.get('steps', []):
            if recovered:
                break
            action = step.get('action')
            if not isinstance(action, Mapping):
                continue
            name = str(action['tool_name'])
            observation, verdict, info, done = env.step(name, dict(action.get('params') or {}))
            verdict = str(getattr(verdict, 'value', verdict))
            info = dict(info or {})
            applied = bool(info.get('applied', verdict in {'PASS', 'SAFE_PROGRESS'}))
            if name not in _DIAGNOSTIC_TOOLS:
                accepted_actions += int(applied)
                rejected_actions += int(not applied)
            violation_count = int(info.get('violation_count', 0))
            process_reward += float(info.get('process_reward', 0.0))
            recovered = bool(getattr(env, 'recovered', False))
            steps.append({'action': dict(action), 'verdict': verdict, 'applied': applied, 'observation': observation, 'info': info})
            if done:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        recovered = False
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            close()
    return {
        'scenario_id': scenario_id,
        'seed': seed,
        'recovered': recovered,
        'accepted_actions': accepted_actions,
        'rejected_actions': rejected_actions,
        'violation_count': violation_count,
        'process_reward': process_reward,
        'steps': steps,
        'error': error,
    }


def select_candidate(candidates: Sequence[Mapping[str, Any]], env_factory: Callable[[], Any], *, scenario_id: str, seed: int = 0) -> dict[str, Any]:
    """Return the selected index, its replay, and all replays for nonempty input.

    Prefer error-free recovery, fewer remaining violations, then fewer accepted
    actions for recovery or more process reward otherwise. Break remaining ties
    by rejected-action count and original candidate order.
    """
    if not candidates:
        raise ValueError('At least one candidate is required')
    verified = [replay_candidate(candidate, env_factory, scenario_id=scenario_id, seed=seed) for candidate in candidates]

    def rank(index: int) -> tuple:
        row = verified[index]
        return (
            row['error'] is not None,
            not row['recovered'],
            row['violation_count'],
            row['accepted_actions'] if row['recovered'] else -row['process_reward'],
            row['rejected_actions'],
            index,
        )

    selected_index = min(range(len(verified)), key=rank)
    return {'selected_index': selected_index, 'selected': verified[selected_index], 'verified_candidates': verified}
