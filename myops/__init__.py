from . import _core

def __getattr__(name):
    if hasattr(_core, name):
        return getattr(_core, name)
    raise AttributeError(f"module 'myops' has no attribute '{name}'")