"""Define constraint violations, checker outputs, and action verdicts."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class Verdict(str, Enum):
    """Distinguish recovery, admissible partial recovery, rejection, and error."""
    PASS = 'PASS'
    SAFE_PROGRESS = 'SAFE_PROGRESS'
    FAIL = 'FAIL'
    ERROR = 'ERROR'

    def __str__(self) -> str:
        """Return the wire-format verdict value."""
        return self.value

@dataclass(frozen=True)
class Violation:
    """Record one device metric outside a limit, with units and severity."""
    constraint_type: str
    device_type: str
    device_id: Any
    metric: str
    value: float
    limit: float
    unit: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return the violation fields as a plain mapping."""
        return {
            'constraint_type': self.constraint_type,
            'device_type': self.device_type,
            'device_id': self.device_id,
            'metric': self.metric,
            'value': self.value,
            'limit': self.limit,
            'unit': self.unit,
            'severity': self.severity,
            'detail': self.detail,
        }

@dataclass
class CheckResult:
    """Group a named check's pass flag, summary metrics, and violations."""
    checker_name: str
    passed: bool
    summary: dict[str, Any]
    violations: list[Violation] = field(default_factory=list)
# A violation branch identifies a constraint, device type, device ID, and metric.
BranchKey = tuple[str, str, str, str]

@dataclass
class VerificationResult:
    """Describe a proposed action and its checks on an isolated simulator.

    Branch mappings hold violation magnitudes before and after the proposal.
    They describe the proposal, regardless of whether the live grid applies it.
    """
    verdict: Verdict
    action: dict[str, Any]
    check_results: list[CheckResult] = field(default_factory=list)
    action_result: dict[str, Any] | None = None
    solver_converged: bool | None = None
    fail_reason: str | None = None
    baseline_branches: dict[BranchKey, float] | None = None
    post_branches: dict[BranchKey, float] = field(default_factory=dict)

    @property
    def violations(self) -> list[Violation]:
        """Return violations flattened across all check results."""
        return [v for check in self.check_results for v in check.violations]

    def to_dict(self) -> dict[str, Any]:
        """Return serializable fields, joining branch-key components with ::."""

        def serialise_branches(branches: dict[BranchKey, float] | None) -> dict[str, float] | None:
            if branches is None:
                return None
            return {'::'.join(key): float(value) for (key, value) in branches.items()}
        return {
            'verdict': self.verdict.value,
            'action': self.action,
            'action_result': self.action_result,
            'solver_converged': self.solver_converged,
            'fail_reason': self.fail_reason,
            'baseline_branches': serialise_branches(self.baseline_branches),
            'post_branches': serialise_branches(self.post_branches),
            'violations': [v.to_dict() for v in self.violations],
            'checks': {
                check.checker_name: {'passed': check.passed, 'summary': check.summary}
                for check in self.check_results
            },
        }
