from hpc101_infer.config import EngineConfig
from hpc101_infer.scheduler.base import (
    RequestState,
    RequestStatus,
    ScheduledOutput,
    ScheduledRequest,
    Scheduler,
)
from hpc101_infer.scheduler.static_batch import StaticBatchScheduler
from hpc101_infer.scheduler.continuous_batch import ContinuousBatchScheduler
from hpc101_infer.scheduler.rebind_batch import RebindScheduler


def create_scheduler(
    config: EngineConfig,
    *,
    default_stop_token_ids: tuple[int, ...],
) -> Scheduler:
    if config.scheduler_backend == "static_batch":
        return StaticBatchScheduler(
            default_stop_token_ids=default_stop_token_ids,
        )
    if config.scheduler_backend == "continuous_batch":
        return ContinuousBatchScheduler(
            default_stop_token_ids=default_stop_token_ids,
        )
    if config.scheduler_backend == "rebind_batch":
        return RebindScheduler(
            default_stop_token_ids=default_stop_token_ids,
        )
    raise ValueError(f"unsupported scheduler backend: {config.scheduler_backend}")


__all__ = [
    "RequestState",
    "RequestStatus",
    "ScheduledOutput",
    "ScheduledRequest",
    "Scheduler",
    "StaticBatchScheduler",
    "ContinuousBatchScheduler",
    "create_scheduler",
]
