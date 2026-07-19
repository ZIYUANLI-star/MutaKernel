"""Isolated paired execution for reference and candidate callables."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch

from .inputs import clone_call_inputs
from .oracle import OracleConfig, compare_outputs
from .state import (
    RNGSnapshot,
    restore_state_dict,
    snapshot_state_dict,
    strict_sync_state_dict,
)
from .types import (
    ExecutionError,
    PhaseTimings,
    ValidationResult,
    ValidationStatus,
)


@dataclass(frozen=True)
class ExecutionConfig:
    """Operational policy for :class:`ValidationExecutor`."""

    synchronize_state: bool = True
    preserve_module_state: bool = True
    preserve_caller_rng: bool = True
    include_cuda_rng: bool = True
    synchronize_cuda_timing: bool = True
    retain_outputs: bool = False
    compare_input_side_effects: bool = True
    compare_module_state: bool = True


def _milliseconds(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _execution_error(phase: str, exc: BaseException) -> ExecutionError:
    return ExecutionError(
        phase=phase,
        exception_type=type(exc).__name__,
        message=str(exc),
    )


def _is_resource_error(error: ExecutionError) -> bool:
    if error.exception_type in {"MemoryError", "OutOfMemoryError"}:
        return True
    return "out of memory" in error.message.lower()


class ValidationExecutor:
    """Run a reference and candidate from identical state, RNG, and inputs."""

    def __init__(
        self,
        oracle_config: Optional[OracleConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> None:
        self.oracle_config = oracle_config or OracleConfig()
        self.execution_config = execution_config or ExecutionConfig()

    def _cuda_sync(self) -> None:
        if self.execution_config.synchronize_cuda_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _invoke(
        self,
        phase: str,
        function: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Tuple[Any, Optional[ExecutionError], float]:
        start_ns = time.perf_counter_ns()
        output = None
        error = None
        try:
            self._cuda_sync()
            output = function(*args, **kwargs)
            self._cuda_sync()
        except BaseException as exc:  # Record kernel/runtime failures as data.
            error = _execution_error(phase, exc)
            try:
                self._cuda_sync()
            except BaseException:
                pass
        return output, error, _milliseconds(start_ns)

    def validate(
        self,
        reference: Callable[..., Any],
        candidate: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> ValidationResult:
        """Validate one candidate call against a reference call.

        The executor restores the caller's RNG and (by default) both modules'
        original state dicts.  Reference failures and infrastructure failures
        produce ``INCONCLUSIVE`` rather than a false defect report.
        """

        total_start_ns = time.perf_counter_ns()
        timings = PhaseTimings()
        errors: List[ExecutionError] = []
        kwargs = {} if kwargs is None else kwargs
        reference_snapshot = None
        candidate_snapshot = None
        caller_rng = None
        replay_rng = None
        reference_output = None
        candidate_output = None
        oracle_result = None
        reference_invocations = 0
        candidate_invocations = 0
        status = ValidationStatus.INCONCLUSIVE
        reason = "validation did not complete"

        try:
            if self.execution_config.preserve_module_state:
                started = time.perf_counter_ns()
                try:
                    reference_snapshot = snapshot_state_dict(reference)
                    candidate_snapshot = snapshot_state_dict(candidate)
                finally:
                    timings.state_snapshot_ms = _milliseconds(started)

            if self.execution_config.synchronize_state:
                started = time.perf_counter_ns()
                try:
                    strict_sync_state_dict(reference, candidate)
                finally:
                    timings.state_sync_ms = _milliseconds(started)

            started = time.perf_counter_ns()
            try:
                caller_rng = RNGSnapshot.capture(
                    include_cuda=self.execution_config.include_cuda_rng,
                )
                replay_rng = RNGSnapshot(
                    python_state=caller_rng.python_state,
                    numpy_state=caller_rng.numpy_state,
                    torch_cpu_state=caller_rng.torch_cpu_state.clone(),
                    torch_cuda_states=None if caller_rng.torch_cuda_states is None else tuple(
                        state.clone() for state in caller_rng.torch_cuda_states
                    ),
                )
            finally:
                timings.rng_capture_ms = _milliseconds(started)

            started = time.perf_counter_ns()
            try:
                reference_args, reference_kwargs = clone_call_inputs(tuple(args), kwargs)
                candidate_args, candidate_kwargs = clone_call_inputs(tuple(args), kwargs)
            finally:
                timings.input_isolation_ms = _milliseconds(started)

            replay_rng.restore()
            reference_invocations += 1
            reference_output, reference_error, timings.reference_ms = self._invoke(
                "reference",
                reference,
                reference_args,
                reference_kwargs,
            )
            if reference_error is not None:
                errors.append(reference_error)

            replay_rng.restore()
            candidate_invocations += 1
            candidate_output, candidate_error, timings.candidate_ms = self._invoke(
                "candidate",
                candidate,
                candidate_args,
                candidate_kwargs,
            )
            if candidate_error is not None:
                errors.append(candidate_error)

            if reference_error is not None:
                status = ValidationStatus.INCONCLUSIVE
                reason = "reference execution failed; candidate correctness is undecidable"
            elif candidate_error is not None:
                if _is_resource_error(candidate_error):
                    status = ValidationStatus.INCONCLUSIVE
                    reason = "candidate execution hit a resource/infrastructure limit"
                else:
                    status = ValidationStatus.FAIL
                    reason = "candidate execution failed while the reference succeeded"
            else:
                started = time.perf_counter_ns()
                reference_observation = {"output": reference_output}
                candidate_observation = {"output": candidate_output}
                if self.execution_config.compare_input_side_effects:
                    reference_observation["post_call_inputs"] = {
                        "args": reference_args,
                        "kwargs": reference_kwargs,
                    }
                    candidate_observation["post_call_inputs"] = {
                        "args": candidate_args,
                        "kwargs": candidate_kwargs,
                    }
                if self.execution_config.compare_module_state:
                    reference_observation["post_call_state"] = snapshot_state_dict(
                        reference
                    )
                    candidate_observation["post_call_state"] = snapshot_state_dict(
                        candidate
                    )
                oracle_result = compare_outputs(
                    reference_observation,
                    candidate_observation,
                    self.oracle_config,
                )
                timings.oracle_ms = _milliseconds(started)
                status = oracle_result.status
                reason = oracle_result.reason

        except BaseException as exc:
            errors.append(_execution_error("setup", exc))
            status = ValidationStatus.INCONCLUSIVE
            reason = f"validation setup failed: {type(exc).__name__}: {exc}"
        finally:
            cleanup_started = time.perf_counter_ns()
            cleanup_errors: List[ExecutionError] = []
            if self.execution_config.preserve_caller_rng and caller_rng is not None:
                try:
                    caller_rng.restore()
                except BaseException as exc:
                    cleanup_errors.append(_execution_error("rng_restore", exc))
            if self.execution_config.preserve_module_state:
                for phase, module, snapshot in (
                    ("reference_state_restore", reference, reference_snapshot),
                    ("candidate_state_restore", candidate, candidate_snapshot),
                ):
                    try:
                        restore_state_dict(module, snapshot)
                    except BaseException as exc:
                        cleanup_errors.append(_execution_error(phase, exc))
            errors.extend(cleanup_errors)
            timings.cleanup_ms = _milliseconds(cleanup_started)
            if cleanup_errors and status is ValidationStatus.PASS:
                status = ValidationStatus.INCONCLUSIVE
                reason = "outputs agreed but validator state cleanup failed"
            timings.total_ms = _milliseconds(total_start_ns)

        return ValidationResult(
            status=status,
            reason=reason,
            timings=timings,
            oracle=oracle_result,
            errors=errors,
            reference_output=(
                reference_output if self.execution_config.retain_outputs else None
            ),
            candidate_output=(
                candidate_output if self.execution_config.retain_outputs else None
            ),
            reference_invocations=reference_invocations,
            candidate_invocations=candidate_invocations,
        )


def validate_pair(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    oracle_config: Optional[OracleConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
) -> ValidationResult:
    """Convenience wrapper around :class:`ValidationExecutor`."""

    return ValidationExecutor(
        oracle_config=oracle_config,
        execution_config=execution_config,
    ).validate(reference, candidate, args=args, kwargs=kwargs)
