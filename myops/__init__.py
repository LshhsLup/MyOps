from . import _backend

def __getattr__(name):
    if hasattr(_backend, name):
        return getattr(_backend, name)
    raise AttributeError(f"module 'myops' has no attribute '{name}'")