"""Verify proposed ANM controls on shadow copies under configurable gates."""

from __future__ import annotations
import math
from typing import Any, Iterable
from .checkers import DEFAULT_CHECKERS
from .tools import ALLOWED_ACTIONS, ToolError, apply_tool
from .types import BranchKey, CheckResult, Verdict, VerificationResult

def _violation_severity(violation: Any) -> float:
    try:
        value = float(violation.value)
        limit = float(violation.limit)
    except (TypeError, ValueError):
        return 1.0
    if not (math.isfinite(value) and math.isfinite(limit)):
        return 1000000.0
    return abs(value - limit)

def violation_branches(checks: Iterable[CheckResult]) -> dict[BranchKey, float]:
    """Map constraint/device/metric keys to their largest absolute excess.

    Non-finite values use a finite sentinel magnitude of one million.
    """
    branches: dict[BranchKey, float] = {}
    for check in checks:
        for violation in check.violations:
            key = (violation.constraint_type, violation.device_type, str(violation.device_id), violation.metric)
            severity = _violation_severity(violation)
            branches[key] = max(branches.get(key, 0.0), severity)
    return branches

def check_manager(manager: Any, checkers: Iterable[Any]=DEFAULT_CHECKERS) -> list[CheckResult]:
    """Run checkers on manager.system_state using manager.base_mva."""
    return [checker.check(manager.system_state, manager.base_mva) for checker in checkers]

class ANMVerifier:
    """Classify tool proposals without modifying the live manager.

    terminal requires all checks to pass. progress permits only existing
    violation keys. progress_mag also bounds each surviving key's magnitude.
    """

    def __init__(self, manager: Any, *, gating_policy: str='progress_mag', checkers: Iterable[Any] | None=None, magnitude_relative_slack: float=0.05, magnitude_abs_floor: float=0.001) -> None:
        """Bind a manager, checks, and relative/absolute magnitude tolerances."""
        if gating_policy not in {'terminal', 'progress', 'progress_mag'}:
            raise ValueError('gating_policy must be terminal, progress, or progress_mag')
        self.manager = manager
        self.gating_policy = gating_policy
        self.checkers = tuple(checkers or DEFAULT_CHECKERS)
        self.magnitude_relative_slack = float(magnitude_relative_slack)
        self.magnitude_abs_floor = float(magnitude_abs_floor)

    def verify(self, action: dict[str, Any]) -> VerificationResult:
        """Return a verdict for a tool_name/params mapping after a shadow solve.

        Unknown tools or invalid parameters return ERROR. Nonconvergence and
        gate violations return FAIL; a constraint-free result returns PASS.
        """
        tool_name = action.get('tool_name', '')
        params = action.get('params', {})
        if tool_name not in ALLOWED_ACTIONS:
            return VerificationResult(verdict=Verdict.ERROR, action=action, fail_reason=f'Action {tool_name!r} is not allowed; available={sorted(ALLOWED_ACTIONS)}')
        shadow = self.manager.create_shadow_copy()
        baseline_checks = check_manager(shadow, self.checkers)
        baseline = violation_branches(baseline_checks)
        try:
            action_result = apply_tool(shadow, tool_name, params)
        except ToolError as exc:
            return VerificationResult(verdict=Verdict.ERROR, action=action, action_result={'status': 'error', 'error': str(exc)}, fail_reason=str(exc), baseline_branches=baseline)
        if not shadow.solve():
            return VerificationResult(verdict=Verdict.FAIL, action=action, action_result={'status': 'success', 'data': action_result}, solver_converged=False, fail_reason='Steady-state solver did not converge after action', baseline_branches=baseline)
        post_checks = check_manager(shadow, self.checkers)
        post = violation_branches(post_checks)
        if all((check.passed for check in post_checks)):
            verdict = Verdict.PASS
            reason = None
        elif self.gating_policy == 'terminal':
            verdict = Verdict.FAIL
            failed = [check.checker_name for check in post_checks if not check.passed]
            reason = f"Constraint violations remain: {', '.join(failed)}"
        else:
            new_branches = [key for key in post if key not in baseline]
            if new_branches:
                verdict = Verdict.FAIL
                reason = f'Violation support expanded by {len(new_branches)} branch(es): {[list(key) for key in new_branches[:3]]}'
            elif self.gating_policy == 'progress_mag':
                drifted: list[tuple[BranchKey, float, float, float]] = []
                for (key, post_severity) in post.items():
                    base_severity = baseline[key]
                    # Use the larger tolerance to avoid tiny-baseline sensitivity.
                    ceiling = max(
                        base_severity * (1.0 + self.magnitude_relative_slack),
                        base_severity + self.magnitude_abs_floor,
                    )
                    if post_severity > ceiling:
                        drifted.append((key, base_severity, post_severity, ceiling))
                if drifted:
                    (key, before, after, ceiling) = max(drifted, key=lambda item: item[2] - item[3])
                    verdict = Verdict.FAIL
                    reason = f'Per-branch magnitude drift on {key}: {before:.4f} -> {after:.4f} exceeds {ceiling:.4f}'
                else:
                    verdict = Verdict.SAFE_PROGRESS
                    reason = f'Admissible recovery step: support {len(baseline)} -> {len(post)}, severities within envelope'
            else:
                verdict = Verdict.SAFE_PROGRESS
                reason = f'Admissible recovery step: post support {len(post)} is a subset of baseline support {len(baseline)}'
        return VerificationResult(verdict=verdict, action=action, check_results=post_checks, action_result={'status': 'success', 'data': action_result}, solver_converged=True, fail_reason=reason, baseline_branches=baseline, post_branches=post)
