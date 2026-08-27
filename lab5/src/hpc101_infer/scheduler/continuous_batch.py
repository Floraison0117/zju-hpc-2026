from __future__ import annotations

from hpc101_infer.scheduler.base import (
    RequestState,
    RequestStatus,
    ScheduledOutput,
    ScheduledRequest,
)


class ContinuousBatchScheduler:
    """Decode scheduler that compacts completed requests out of active slots.

    The engine keeps a fixed physical KV-cache allocation, while this scheduler
    marks completed requests inactive immediately.  Inactive slots are masked
    in the next decode step, so finished requests no longer consume decode work.
    """

    def __init__(self, default_stop_token_ids: tuple[int, ...]) -> None:
        self.default_stop_token_ids = default_stop_token_ids
        self.requests: list[RequestState] = []
        self._prefill_scheduled = False
        self.active_counts: list[int] = []

    def add_request(self, request: RequestState) -> None:
        if self._prefill_scheduled:
            raise RuntimeError("cannot add requests after prefill starts")
        if request.status is not RequestStatus.PENDING:
            raise ValueError("new requests must be pending")
        self.requests.append(request)

    def schedule(self) -> ScheduledOutput:
        if not self.requests:
            raise RuntimeError("cannot schedule an empty batch")
        if not self._prefill_scheduled:
            self._prefill_scheduled = True
            scheduled = []
            for state in self.requests:
                state.num_computed_tokens = len(state.prompt_token_ids)
                if state.request.max_new_tokens == 0:
                    state.status = RequestStatus.COMPLETED
                    state.finish_reason = "length"
                else:
                    state.status = RequestStatus.PREFILLING
                scheduled.append(ScheduledRequest(state, len(state.prompt_token_ids)))
            self.active_counts.append(sum(s.request.status is not RequestStatus.COMPLETED for s in scheduled))
            return ScheduledOutput("prefill", scheduled)

        if not self.has_unfinished_requests():
            raise RuntimeError("cannot schedule a completed batch")
        scheduled = []
        for state in self.requests:
            active = state.status is RequestStatus.DECODING
            count = int(active)
            state.num_computed_tokens += count
            scheduled.append(ScheduledRequest(state, count))
        self.active_counts.append(sum(item.num_scheduled_tokens > 0 for item in scheduled))
        return ScheduledOutput("decode", scheduled)

    def update(self, token_ids: list[int]) -> None:
        if len(token_ids) != len(self.requests):
            raise ValueError("token_ids must match the physical batch size")
        for state, token_id in zip(self.requests, token_ids, strict=True):
            if state.status not in (RequestStatus.PREFILLING, RequestStatus.DECODING):
                continue
            state.output_token_ids.append(token_id)
            stops = self.default_stop_token_ids if state.request.stop_token_ids is None else state.request.stop_token_ids
            if token_id in stops:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "stop"
            elif len(state.output_token_ids) >= state.request.max_new_tokens:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "length"
            else:
                state.status = RequestStatus.DECODING

    def has_unfinished_requests(self) -> bool:
        return any(state.status not in (RequestStatus.COMPLETED, RequestStatus.FAILED) for state in self.requests)
