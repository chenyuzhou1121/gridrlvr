"""Expose simulator environments and restoration rewards through lazy imports."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["ANMEnv", "Verdict", "RewardConfig", "outcome_reward", "process_reward", "total_reward"]

def __getattr__(name):
    """Resolve a public environment or reward symbol on first access."""
    if name in {"ANMEnv", "Verdict"}:
        return getattr(import_module(".env", __name__), name)
    if name in __all__:
        return getattr(import_module(".reward", __name__), name)
    raise AttributeError(name)
