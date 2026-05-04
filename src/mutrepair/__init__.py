def __getattr__(name):
    if name == "EnhancedInputGenerator":
        from .enhanced_inputs import EnhancedInputGenerator
        return EnhancedInputGenerator
    if name == "FeedbackBuilder":
        from .feedback_builder import FeedbackBuilder
        return FeedbackBuilder
    if name == "RepairLoop":
        from .repair_loop import RepairLoop
        return RepairLoop
    if name == "ExperienceStore":
        from .experience_store import ExperienceStore
        return ExperienceStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
