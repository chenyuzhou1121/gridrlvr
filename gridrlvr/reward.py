"""Compute outcome, constraint, and incremental restoration rewards."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RewardConfig:
    """Set verdict penalties and weights for outcome and process components.

    Penalty values are signed rewards. step_cost is subtracted on each call;
    outcome_weight applies only when reward_breakdown is called as terminal.
    """
    outcome_success: float = 1.0
    outcome_failure: float = 0.0
    warning_penalty: float = -0.3
    violation_penalty: float = -0.6
    critical_penalty: float = -1.0
    error_penalty: float = -1.0
    support_weight: float = 0.6
    severity_weight: float = 0.3
    drift_weight: float = 0.3
    process_enabled: bool = True
    outcome_weight: float = 1.0
    process_weight: float = 1.0
    violation_weight: float = 1.0
    step_cost: float = 0.0

def outcome_reward(recovered: bool, config: RewardConfig | None=None) -> float:
    """Return the configured success or failure reward for a recovery flag."""
    cfg = config or RewardConfig()
    return cfg.outcome_success if bool(recovered) else cfg.outcome_failure

def violation_reward(result: Any, config: RewardConfig | None=None) -> float:
    """Return the most negative violation penalty, or the ERROR penalty.

    result supplies verdict and check_results with violation severities.
    Non-failing verdicts receive zero; FAIL without violations uses warning.
    """
    cfg = config or RewardConfig()
    verdict = _verdict_value(result.verdict)
    if verdict == 'ERROR':
        return cfg.error_penalty
    if verdict != 'FAIL':
        return 0.0
    by_severity = {'warning': cfg.warning_penalty, 'violation': cfg.violation_penalty, 'critical': cfg.critical_penalty}
    penalties = [by_severity.get(violation.severity, cfg.violation_penalty) for check in result.check_results for violation in check.violations]
    return min(penalties, default=cfg.warning_penalty)

def violation_penalty(result: Any, config: RewardConfig | None=None) -> float:
    """Return violation_reward with the same signed-penalty convention."""
    return violation_reward(result, config)

def process_reward(result: Any, config: RewardConfig | None=None) -> float:
    """Score SAFE_PROGRESS from baseline and proposed violation magnitudes.

    Average eliminated support and severity reduction across constraint
    families, then subtract the largest surviving relative drift, capped at one.
    Missing baselines and other verdicts receive zero.
    """
    cfg = config or RewardConfig()
    if not cfg.process_enabled:
        return 0.0
    if _verdict_value(result.verdict) != 'SAFE_PROGRESS':
        return 0.0
    before = result.baseline_branches
    after = result.post_branches or {}
    if not before:
        return 0.0
    before_keys = set(before)
    after_keys = set(after)
    eliminated = before_keys - after_keys
    surviving = before_keys & after_keys
    families: dict[str, list[Any]] = {}
    for key in before_keys:
        family = key[0] if isinstance(key, tuple) else str(key).split('::', 1)[0]
        families.setdefault(family, []).append(key)
    support_terms: list[float] = []
    severity_terms: list[float] = []
    drift_terms: list[float] = []
    for keys in families.values():
        # Normalize within each family so metric units do not mix in its sum.
        family_total = sum((float(before[key]) for key in keys)) + 1e-08
        support_terms.append(sum((float(before[key]) for key in keys if key in eliminated)) / family_total)
        severity_terms.append(sum((max(0.0, float(before[key]) - float(after[key])) for key in keys if key in surviving)) / family_total)
        drift_terms.extend((max(0.0, float(after[key]) - float(before[key])) / (float(before[key]) + 1e-08) for key in keys if key in surviving))
    support_elim = sum(support_terms) / len(support_terms)
    severity_reduction = sum(severity_terms) / len(severity_terms)
    drift = max(drift_terms, default=0.0)
    return cfg.support_weight * support_elim + cfg.severity_weight * severity_reduction - cfg.drift_weight * min(drift, 1.0)

def reward_breakdown(result: Any, *, recovered: bool, terminal: bool=False, config: RewardConfig | None=None) -> dict[str, float]:
    """Return unweighted components and their weighted total for one step.

    result is a verification result; the outcome component is zero until
    terminal is true. step_cost is included only in total_reward.
    """
    cfg = config or RewardConfig()
    outcome = outcome_reward(recovered, cfg) if terminal else 0.0
    violation = violation_reward(result, cfg)
    process = process_reward(result, cfg)
    total = cfg.outcome_weight * outcome + cfg.violation_weight * violation + cfg.process_weight * process - cfg.step_cost
    return {
        'outcome_reward': float(outcome),
        'violation_reward': float(violation),
        'process_reward': float(process),
        'total_reward': float(total),
    }

def total_reward(result: Any, *, recovered: bool, terminal: bool=False, config: RewardConfig | None=None) -> float:
    """Return the weighted scalar from reward_breakdown for one step."""
    return reward_breakdown(result, recovered=recovered, terminal=terminal, config=config)['total_reward']
compute_reward = total_reward

def _verdict_value(value: Any) -> str:
    return str(getattr(value, 'value', value))
__all__ = ['RewardConfig', 'outcome_reward', 'violation_penalty', 'violation_reward', 'process_reward', 'reward_breakdown', 'total_reward', 'compute_reward']
