def __getattr__(name):
    if name == "KernelBenchBridge":
        from .eval_bridge import KernelBenchBridge
        return KernelBenchBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
