"""Score simulator trajectories supplied through verl reward extra_info."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'recovered', 'pass'}
    return bool(value)

def _finite_float(value: Any, default: float=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float('inf'), float('-inf')):
        return default
    return result

def _step_process_reward(step: Mapping[str, Any]) -> float:
    for container in (step, step.get('info', {})):
        if not isinstance(container, Mapping):
            continue
        for key in ('process_reward', 'step_reward', 'reward'):
            if key in container:
                return _finite_float(container[key])
    return 0.0

def _step_violations(step: Mapping[str, Any]) -> int:
    verdict = str(step.get('verdict', '')).upper()
    return int(verdict.endswith('FAIL') or verdict.endswith('ERROR'))

def trajectory_reward(*, recovered: bool, steps: Sequence[Mapping[str, Any]]=(), outcome_weight: float=1.0, process_reward_weight: float=0.0, violation_penalty: float=0.0) -> dict[str, float | int | bool]:
    """Return a weighted recovery score and trajectory component totals.

    Process values come from each step or its info mapping. FAIL/ERROR verdicts
    count as violations; violation_penalty is subtracted per such verdict.
    """
    outcome_reward = float(_as_bool(recovered))
    process_reward = sum((_step_process_reward(step) for step in steps))
    violation_count = sum((_step_violations(step) for step in steps))
    score = _finite_float(outcome_weight, 1.0) * outcome_reward + _finite_float(process_reward_weight) * process_reward - _finite_float(violation_penalty) * violation_count
    return {'score': float(score), 'outcome_reward': outcome_reward, 'process_reward': float(process_reward), 'violation_count': int(violation_count), 'recovered': bool(recovered)}

def _trajectory_from_extra(extra_info: Mapping[str, Any]) -> tuple[bool, Sequence[Mapping[str, Any]]]:
    candidates = [extra_info]
    tool_fields = extra_info.get('tool_extra_fields')
    if isinstance(tool_fields, Mapping):
        candidates.append(tool_fields)
    for candidate in candidates:
        trajectory = candidate.get('anm_trajectory', candidate.get('trajectory', []))
        recovered = candidate.get('recovered')
        if isinstance(trajectory, Mapping):
            if recovered is None:
                recovered = trajectory.get('recovered')
            trajectory = trajectory.get('steps', [])
        if isinstance(trajectory, Sequence) and (not isinstance(trajectory, (str, bytes, bytearray))):
            steps = [step for step in trajectory if isinstance(step, Mapping)]
            if recovered is not None or steps:
                return (_as_bool(recovered), steps)
    return (False, [])

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Mapping[str, Any] | None=None, *, outcome_weight: float=1.0, process_reward_weight: float=0.0, violation_penalty: float=0.0) -> dict[str, float | int | bool]:
    """Score simulator metadata in extra_info; ignore generated answer text.

    Accept anm_trajectory or trajectory at the top level or in tool_extra_fields.
    Recovery is supplied by metadata, never inferred from solution_str.
    """
    del data_source, solution_str, ground_truth
    (recovered, steps) = _trajectory_from_extra(extra_info or {})
    return trajectory_reward(recovered=recovered, steps=steps, outcome_weight=outcome_weight, process_reward_weight=process_reward_weight, violation_penalty=violation_penalty)


def compute_score_restorebench(data_source: str, solution_str: str, ground_truth: Any, extra_info: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Delegate the verl reward signature to the RestoreBench trajectory scorer."""
    from gridrlvr.verl_adapter.restorebench_loop import compute_score_restorebench as score
    return score(data_source, solution_str, ground_truth, extra_info, **kwargs)
