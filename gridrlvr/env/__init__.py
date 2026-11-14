"""Expose ANM environment components without eagerly loading simulators."""

from importlib import import_module

_EXPORTS = {
    'ANMEnv': ('anm_env', 'ANMEnv'),
    'ANMBranchLoadingChecker': ('checkers', 'ANMBranchLoadingChecker'),
    'ANMStorageSoCChecker': ('checkers', 'ANMStorageSoCChecker'),
    'ANMVoltageChecker': ('checkers', 'ANMVoltageChecker'),
    'GymANMManager': ('manager', 'GymANMManager'),
    'get_mpc_action': ('mpc', 'get_mpc_action'),
    'run_mpc': ('mpc', 'run_mpc'),
    'ANMScenario': ('scenarios', 'ANMScenario'),
    'ScenarioCatalog': ('scenarios', 'ScenarioCatalog'),
    'apply_limits': ('scenarios', 'apply_limits'),
    'derive_env_seed': ('scenarios', 'derive_env_seed'),
    'encode_limit_suffix': ('scenarios', 'encode_limit_suffix'),
    'normalize_limits': ('scenarios', 'normalize_limits'),
    'parse_limit_suffix': ('scenarios', 'parse_limit_suffix'),
    'with_limits': ('scenarios', 'with_limits'),
    'ALLOWED_ACTIONS': ('tools', 'ALLOWED_ACTIONS'),
    'build_tool_schemas': ('tools', 'build_tool_schemas'),
    'CheckResult': ('types', 'CheckResult'),
    'Verdict': ('types', 'Verdict'),
    'VerificationResult': ('types', 'VerificationResult'),
    'Violation': ('types', 'Violation'),
    'ANMVerifier': ('verifier', 'ANMVerifier'),
    'parse_scenario_id': ('scenarios', 'parse_scenario_id'),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    """Load and cache a named public component, or raise AttributeError."""
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module("." + module, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(name)
