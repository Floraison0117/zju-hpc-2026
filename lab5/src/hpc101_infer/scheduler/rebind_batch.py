"""Slot-rebinding scheduler: first-call prefill, subsequent decode with admission."""

from __future__ import annotations
from hpc101_infer.scheduler.base import RequestState, RequestStatus, ScheduledOutput, ScheduledRequest


class RebindScheduler:
    def __init__(self, default_stop_token_ids: tuple[int, ...]) -> None:
        self.default_stop_token_ids = default_stop_token_ids
        self._pending: list[RequestState] = []
        self._active_states: list[RequestState] = []
        self._completed: list[RequestState] = []
        self._prefill_done = False

    def add_request(self, request: RequestState) -> None:
        if request.status is not RequestStatus.PENDING:
            raise ValueError("new requests must be pending")
        self._pending.append(request)

    def schedule(self) -> ScheduledOutput:
        if not self._prefill_done and self._pending:
            self._prefill_done = True
            state = self._pending.pop(0)
            state.num_computed_tokens = len(state.prompt_token_ids)
            if state.request.max_new_tokens == 0:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "length"
                self._completed.append(state)
            else:
                state.status = RequestStatus.PREFILLING
                self._active_states.append(state)
            return ScheduledOutput("prefill", [ScheduledRequest(state, len(state.prompt_token_ids))])

        scheduled = []
        for state in self._active_states:
            if state.status in (RequestStatus.PREFILLING, RequestStatus.DECODING):
                state.num_computed_tokens += 1
                scheduled.append(ScheduledRequest(state, 1))
            else:
                scheduled.append(ScheduledRequest(state, 0))
        return ScheduledOutput("decode", scheduled)

    def update(self, token_ids: list[int]) -> None:
        for state, tok in zip(self._active_states, token_ids):
            if state.status not in (RequestStatus.PREFILLING, RequestStatus.DECODING):
                continue
            state.output_token_ids.append(tok)
            stops = self.default_stop_token_ids if state.request.stop_token_ids is None else state.request.stop_token_ids
            if tok in stops:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "stop"
            elif len(state.output_token_ids) >= state.request.max_new_tokens:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "length"
            else:
                state.status = RequestStatus.DECODING

    def has_unfinished_requests(self) -> bool:
        if self._pending:
            return True
        return any(s.status not in (RequestStatus.COMPLETED, RequestStatus.FAILED) for s in self._active_states)
