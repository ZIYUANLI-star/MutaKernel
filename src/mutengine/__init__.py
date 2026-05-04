# Lazy imports to avoid requiring torch at parse time
def __getattr__(name):
    if name == "MutantRunner":
        from .mutant_runner import MutantRunner
        return MutantRunner
    if name == "EquivalentDetector":
        from .equivalent_detector import EquivalentDetector
        return EquivalentDetector
    if name == "MutationReporter":
        from .report import MutationReporter
        return MutationReporter
    if name == "CompareResult":
        from .equivalent_detector import CompareResult
        return CompareResult
    if name == "check_all_rules":
        from .static_equiv_rules import check_all_rules
        return check_all_rules
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
