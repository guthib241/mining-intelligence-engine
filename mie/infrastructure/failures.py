"""Failure taxonomy (Missing-Data and Failure Protocol). A failed run is never a scientific rejection."""
class MIEFailure(Exception):
    kind = "UNKNOWN"
class CodeFailure(MIEFailure):
    kind = "CODE_FAILURE"
class DataFailure(MIEFailure):
    kind = "DATA_FAILURE"
class EnvironmentFailure(MIEFailure):
    kind = "ENVIRONMENT_FAILURE"
class ExperimentFailure(MIEFailure):
    kind = "EXPERIMENT_FAILURE"
class ModelFailure(MIEFailure):
    kind = "MODEL_FAILURE"
class HypothesisFailure(MIEFailure):
    kind = "HYPOTHESIS_FAILURE"
class Blocked(MIEFailure):
    kind = "BLOCKED"

def classify_exception(e: BaseException) -> str:
    if isinstance(e, MIEFailure):
        return e.kind
    if isinstance(e, FileNotFoundError):
        return "DATA_FAILURE"
    if isinstance(e, KeyError):
        return "EXPERIMENT_FAILURE"   # bad config / missing feature column
    if isinstance(e, (MemoryError, TimeoutError, OSError, ImportError)):
        return "ENVIRONMENT_FAILURE"
    return "CODE_FAILURE"
