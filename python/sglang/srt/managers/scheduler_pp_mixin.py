from __future__ import annotations

import logging
import math
import os
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj

logger = logging.getLogger(__name__)

_PP_REQ_PAYLOAD_V1 = "__pp_req_payload_v1__"
_PP_RELEASE_PAYLOAD_V1 = "__pp_release_payload_v1__"
_PP_RELEASE_PAYLOAD_V2 = "__pp_release_payload_v2__"
_PP_FRONTIER_ACK_HEADER = struct.Struct("<qI?")
_PP_FRONTIER_ACK_LEN = struct.Struct("<I")


class _PPPrefillDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if os.getenv("SGLANG_DEBUG_HICACHE_VERBOSE", "0") == "1":
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not (
            message.startswith("[PPReqPhase]")
            or message.startswith("[PP Dynamic Chunk]")
        )


if not any(isinstance(f, _PPPrefillDebugFilter) for f in logger.filters):
    logger.addFilter(_PPPrefillDebugFilter())

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _ordered_intersection(left: List[str], right: List[str]) -> List[str]:
    right_set = set(right)
    return [item for item in left if item in right_set]


def _ordered_common_prefix(left: List[str], right: List[str]) -> List[str]:
    prefix = []
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        prefix.append(left_item)
    return prefix


def _ordered_prefix_from_queue(queue_rids: List[str], proposal_rids: List[str]) -> List[str]:
    """Keep only the authoritative prefix that still matches the local queue head."""
    prefix = []
    for queue_item, proposal_item in zip(queue_rids, proposal_rids):
        if queue_item != proposal_item:
            break
        prefix.append(proposal_item)
    return prefix


def _ordered_union(left: List[str], right: List[str]) -> List[str]:
    merged = list(left)
    seen = set(left)
    for item in right:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


class SchedulerPPMixin:
    def _pp_encode_frontier_ack_blob(
        self: Scheduler,
        mb_id: int,
        ack_rids,
        barrier_rid: Optional[str],
    ) -> bytes:
        ack_rids = tuple(ack_rids or ())
        out = bytearray(
            _PP_FRONTIER_ACK_HEADER.pack(
                int(mb_id),
                len(ack_rids),
                barrier_rid is not None,
            )
        )
        for rid in ack_rids:
            rid_bytes = str(rid).encode("utf-8")
            out.extend(_PP_FRONTIER_ACK_LEN.pack(len(rid_bytes)))
            out.extend(rid_bytes)
        if barrier_rid is not None:
            barrier_bytes = str(barrier_rid).encode("utf-8")
            out.extend(_PP_FRONTIER_ACK_LEN.pack(len(barrier_bytes)))
            out.extend(barrier_bytes)
        return bytes(out)

    def _pp_decode_frontier_ack_blob(
        self: Scheduler, blob: bytes
    ) -> Tuple[Optional[int], tuple[str, ...], Optional[str]]:
        if not blob:
            return None, (), None
        view = memoryview(blob)
        if len(view) < _PP_FRONTIER_ACK_HEADER.size:
            raise ValueError("launch_frontier_ack blob truncated")
        mb_id, ack_count, has_barrier = _PP_FRONTIER_ACK_HEADER.unpack_from(
            view, 0
        )
        offset = _PP_FRONTIER_ACK_HEADER.size
        ack_rids = []
        for _ in range(int(ack_count)):
            if len(view) < offset + _PP_FRONTIER_ACK_LEN.size:
                raise ValueError("launch_frontier_ack rid length truncated")
            rid_len = _PP_FRONTIER_ACK_LEN.unpack_from(view, offset)[0]
            offset += _PP_FRONTIER_ACK_LEN.size
            end = offset + rid_len
            if len(view) < end:
                raise ValueError("launch_frontier_ack rid truncated")
            ack_rids.append(view[offset:end].tobytes().decode("utf-8"))
            offset = end
        barrier_rid = None
        if has_barrier:
            if len(view) < offset + _PP_FRONTIER_ACK_LEN.size:
                raise ValueError("launch_frontier_ack barrier length truncated")
            barrier_len = _PP_FRONTIER_ACK_LEN.unpack_from(view, offset)[0]
            offset += _PP_FRONTIER_ACK_LEN.size
            end = offset + barrier_len
            if len(view) < end:
                raise ValueError("launch_frontier_ack barrier truncated")
            barrier_rid = view[offset:end].tobytes().decode("utf-8")
            offset = end
        if offset != len(view):
            raise ValueError("launch_frontier_ack blob has trailing bytes")
        return int(mb_id), tuple(ack_rids), barrier_rid

    def _pp_pack_req_payload(
        self: Scheduler, recv_reqs, hicache_host_tree_events: List[object]
    ):
        if not hicache_host_tree_events:
            return recv_reqs
        return (
            _PP_REQ_PAYLOAD_V1,
            recv_reqs,
            hicache_host_tree_events,
        )

    def _pp_unpack_req_payload(
        self: Scheduler, payload
    ) -> Tuple[object, List[object]]:
        if (
            isinstance(payload, tuple)
            and len(payload) == 3
            and payload[0] == _PP_REQ_PAYLOAD_V1
        ):
            return payload[1], payload[2] or ()
        if isinstance(payload, dict) and "recv_reqs" in payload:
            return payload["recv_reqs"], payload.get("hicache_host_tree_events") or ()
        return payload, ()

    def _pp_prefill_safe_len(self: Scheduler, value) -> int:
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    def _pp_prefill_diag_enabled(self: Scheduler) -> bool:
        return os.getenv("SGLANG_DEBUG_PP_PREFILL_DIAG", "0") == "1"

    def _pp_prefill_diag_rids(
        self: Scheduler, items, limit: int = 8
    ) -> List[str]:
        if not items:
            return []
        out = []
        for item in list(items)[:limit]:
            out.append(getattr(item, "rid", item))
        return out

    def _pp_prefill_diag_queue(
        self: Scheduler, items, limit: int = 8
    ) -> List[str]:
        return self._pp_prefill_diag_rids(items, limit=limit)

    def _pp_prefill_diag_log(self: Scheduler, tag: str, **kwargs) -> None:
        if not self._pp_prefill_diag_enabled():
            return
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        logger.warning(
            "[PPPrefillDiag][%s] pp=%s cp=%s tp=%s %s",
            tag,
            self.pp_rank,
            self.attn_cp_rank,
            self.attn_tp_rank,
            " ".join(parts),
        )

    def _pp_prefill_req_shape(
        self: Scheduler, req: Req
    ) -> Dict[str, object]:
        return {
            "rid": req.rid,
            "fill": self._pp_prefill_safe_len(getattr(req, "fill_ids", None)),
            "prefix": self._pp_prefill_safe_len(
                getattr(req, "prefix_indices", None)
            ),
            "storage": getattr(req, "storage_hit_length", 0),
        }

    def _pp_prefill_batch_shape(
        self: Scheduler, batch: Optional[ScheduleBatch], limit: int = 6
    ) -> List[Dict[str, object]]:
        if batch is None:
            return []
        return [self._pp_prefill_req_shape(req) for req in batch.reqs[:limit]]

    def _pp_prefill_batch_rids(
        self: Scheduler, batch: Optional[ScheduleBatch], limit: int = 16
    ) -> List[str]:
        if batch is None:
            return []
        return self._pp_prefill_diag_rids(batch.reqs[:limit])

    def _pp_prefill_queue_shape(
        self: Scheduler, queue, limit: int = 6
    ) -> List[Dict[str, object]]:
        if not queue:
            return []
        return [self._pp_prefill_req_shape(req) for req in list(queue)[:limit]]

    def _pp_prefill_bootstrap_req_debug(
        self: Scheduler, reqs, limit: int = 8
    ) -> List[Dict[str, object]]:
        if not reqs:
            return []
        debug_reqs = []
        for req in list(reqs)[:limit]:
            debug_reqs.append(
                {
                    "rid": req.rid,
                    "mb": getattr(req, "bootstrap_mb_id", None),
                    "room": getattr(req, "bootstrap_room", None),
                }
            )
        return debug_reqs

    def _pp_prefill_bootstrap_poll_summary(
        self: Scheduler,
        mb_id: int,
        local_bootstrap_reqs,
        local_candidate_reqs,
        curr_good_bootstrapped_rids,
        curr_bad_bootstrapped_rids,
        candidate_polls=None,
    ) -> Dict[str, object]:
        poll_values = []
        if candidate_polls:
            poll_values = [
                getattr(poll, "name", str(poll))
                for poll in list(candidate_polls)[:8]
            ]
        return {
            "mb": mb_id,
            "bootstrap": self._pp_prefill_bootstrap_req_debug(local_bootstrap_reqs),
            "candidates": self._pp_prefill_bootstrap_req_debug(local_candidate_reqs),
            "polls": poll_values,
            "curr_good": curr_good_bootstrapped_rids[:8],
            "curr_bad": curr_bad_bootstrapped_rids[:8],
        }

    def _pp_prefill_problem_should_log(
        self: Scheduler, tag: str, key: Tuple[object, ...]
    ) -> bool:
        if not self._pp_prefill_diag_enabled():
            return False
        state = getattr(self, "_pp_prefill_problem_state", None)
        if state is None:
            state = {}
            setattr(self, "_pp_prefill_problem_state", state)
        full_key = (tag,) + tuple(key)
        count = state.get(full_key, 0) + 1
        state[full_key] = count
        return count in (1, 8, 64, 256)

    def _pp_prefill_problem_log(
        self: Scheduler, tag: str, key: Tuple[object, ...], **kwargs
    ) -> None:
        if not self._pp_prefill_problem_should_log(tag, key):
            return
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        logger.warning(
            "[PPPrefillProblem][%s] pp=%s cp=%s tp=%s %s",
            tag,
            self.pp_rank,
            self.attn_cp_rank,
            self.attn_tp_rank,
            " ".join(parts),
        )

    def _pp_prefill_intermediate_head_count(
        self: Scheduler, rid: str
    ) -> int:
        state = getattr(self, "_pp_prefill_intermediate_head_state", None)
        if state is None:
            state = {}
            setattr(self, "_pp_prefill_intermediate_head_state", state)
        count = state.get(rid, 0) + 1
        state[rid] = count
        return count

    def _pp_prefill_clear_intermediate_head(
        self: Scheduler, rid: Optional[str]
    ) -> None:
        if rid is None:
            return
        state = getattr(self, "_pp_prefill_intermediate_head_state", None)
        if state is None:
            return
        state.pop(rid, None)

    def _pp_prefill_defer_bootstrap_rids(
        self: Scheduler, deferred_rids: List[str]
    ) -> List[str]:
        if not deferred_rids:
            return []
        queue = self.disagg_prefill_bootstrap_queue.queue
        deferred_index = {rid: idx for idx, rid in enumerate(deferred_rids)}
        deferred_reqs = []
        remaining_reqs = []
        for req in queue:
            if req.rid in deferred_index:
                deferred_reqs.append(req)
            else:
                remaining_reqs.append(req)
        if not deferred_reqs:
            return []
        deferred_reqs.sort(key=lambda req: deferred_index[req.rid])
        self.disagg_prefill_bootstrap_queue.queue = remaining_reqs + deferred_reqs
        for req in deferred_reqs:
            req.bootstrap_mb_id = None
            self._pp_prefill_clear_intermediate_head(req.rid)
            if hasattr(self.tree_cache, "discard_pp_locally_revoked_req"):
                self.tree_cache.discard_pp_locally_revoked_req(req.rid)
            elif hasattr(self.tree_cache, "pp_locally_revoked_req_ids"):
                self.tree_cache.pp_locally_revoked_req_ids.discard(req.rid)
        return [req.rid for req in deferred_reqs]

    def _pp_unpack_bootstrap_payload(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ) -> Tuple[List[str], List[str], List[str], int]:
        if bootstrapped_rids is None:
            return [], [], [], 0
        if len(bootstrapped_rids) == 0:
            return [], [], [], 0
        if len(bootstrapped_rids) == 4:
            good_rids, bad_rids, deferred_rids, shared_capacity = bootstrapped_rids
            return good_rids, bad_rids, deferred_rids, shared_capacity
        good_rids, bad_rids, shared_capacity = bootstrapped_rids
        return good_rids, bad_rids, [], shared_capacity

    def _pp_pack_release_payload(
        self: Scheduler,
        release_rids: Optional[List[str]],
        launch_ack_mb_id: Optional[int] = None,
        launch_ack_rids: Optional[List[str]] = None,
        launch_ack_barrier_rid: Optional[str] = None,
    ):
        if (
            launch_ack_mb_id is None
            or (
                not launch_ack_rids
                and launch_ack_barrier_rid is None
            )
        ):
            return release_rids
        ack_blob = self._pp_encode_frontier_ack_blob(
            launch_ack_mb_id,
            launch_ack_rids,
            launch_ack_barrier_rid,
        )
        return (
            _PP_RELEASE_PAYLOAD_V2,
            list(release_rids or []),
            ack_blob,
        )

    def _pp_unpack_release_payload(
        self: Scheduler, payload
    ) -> Tuple[Optional[List[str]], Optional[int], tuple[str, ...], Optional[str]]:
        if payload is None:
            return None, None, (), None
        if (
            isinstance(payload, tuple)
            and len(payload) == 3
            and payload[0] == _PP_RELEASE_PAYLOAD_V2
        ):
            ack_mb_id, ack_rids, ack_barrier_rid = self._pp_decode_frontier_ack_blob(
                payload[2]
            )
            return payload[1], ack_mb_id, ack_rids, ack_barrier_rid
        if (
            isinstance(payload, tuple)
            and len(payload) == 5
            and payload[0] == _PP_RELEASE_PAYLOAD_V1
        ):
            return payload[1], payload[2], tuple(payload[3] or ()), payload[4]
        if isinstance(payload, dict):
            release_rids = payload.get("release_rids")
            ack_payload = payload.get("launch_frontier_ack") or {}
            ack_mb_id = ack_payload.get("mb_id")
            ack_rids = tuple(ack_payload.get("rids") or ())
            ack_barrier_rid = ack_payload.get("barrier_rid")
            return release_rids, ack_mb_id, ack_rids, ack_barrier_rid
        return payload, None, (), None

    def _pp_record_launch_frontier_ack(
        self: Scheduler,
        ack_mb_id: Optional[int],
        ack_rids,
        ack_barrier_rid: Optional[str] = None,
    ) -> None:
        if ack_mb_id is None or (not ack_rids and ack_barrier_rid is None):
            return
        # Apply launch-frontier acks on the next visit of the same mb_id.
        # This keeps the semantics "PP0 may lead by at most one microbatch"
        # instead of trying to retroactively constrain the microbatch that is
        # already in flight when the downstream ack arrives.
        ack_rids = tuple(ack_rids or ())
        self.pp_pending_launch_frontier_ack_by_mb[ack_mb_id] = {
            "rids": ack_rids,
            "barrier_rid": ack_barrier_rid,
        }
        if hasattr(self, "_record_pp_frontier_ack_recv"):
            self._record_pp_frontier_ack_recv()
        if self._pp_prefill_diag_enabled():
            logger.warning(
                "[PPPrefillDiag][launch_frontier_ack_recv] pp=%s cp=%s tp=%s mb=%s ack=%s barrier=%s",
                self.pp_rank,
                self.attn_cp_rank,
                self.attn_tp_rank,
                ack_mb_id,
                ack_rids[:8],
                ack_barrier_rid,
            )

    def _pp_consume_launch_frontier_ack(
        self: Scheduler, mb_id: Optional[int]
    ) -> Optional[Dict[str, object]]:
        if mb_id is None:
            return None
        ack = self.pp_launch_frontier_ack_by_mb.pop(mb_id, None)
        if ack is None:
            return None
        if hasattr(self, "_record_pp_frontier_ack_consume"):
            self._record_pp_frontier_ack_consume()
        return {
            "rids": ack.get("rids") or (),
            "barrier_rid": ack.get("barrier_rid"),
        }

    def _pp_activate_launch_frontier_ack(self: Scheduler, mb_id: int) -> None:
        pending_ack = self.pp_pending_launch_frontier_ack_by_mb.pop(mb_id, None)
        if pending_ack is None:
            return
        self.pp_launch_frontier_ack_by_mb[mb_id] = pending_ack
        if hasattr(self, "_record_pp_frontier_ack_activate"):
            self._record_pp_frontier_ack_activate()
        if self._pp_prefill_diag_enabled():
            logger.warning(
                "[PPPrefillDiag][launch_frontier_ack_activate] pp=%s cp=%s tp=%s mb=%s ack=%s barrier=%s",
                self.pp_rank,
                self.attn_cp_rank,
                self.attn_tp_rank,
                mb_id,
                list(pending_ack.get("rids") or [])[:8],
                pending_ack.get("barrier_rid"),
            )

    def _pp_prefill_shape_snapshot_log(
        self: Scheduler,
        tag: str,
        key: Tuple[object, ...],
        batch: Optional[ScheduleBatch],
        waiting_queue,
        bootstrap_queue,
        inflight_queue,
    ) -> None:
        self._pp_prefill_problem_log(
            tag,
            key=key,
            batch=self._pp_prefill_batch_shape(batch),
            waiting=self._pp_prefill_queue_shape(waiting_queue),
            bootstrap=self._pp_prefill_queue_shape(bootstrap_queue),
            inflight=self._pp_prefill_diag_rids(inflight_queue),
            prealloc=len(getattr(self, "disagg_prefill_prealloc_queue", [])),
        )

    def _pp_prefill_req_age_sec(
        self: Scheduler, req: Req, field: str
    ) -> float:
        ts = getattr(req.time_stats, field, 0.0)
        if not ts:
            return 0.0
        return max(0.0, time.perf_counter() - ts)

    def _pp_prefill_age_bucket(self: Scheduler, age_sec: float) -> int:
        if age_sec >= 48:
            return 48
        if age_sec >= 32:
            return 32
        if age_sec >= 16:
            return 16
        if age_sec >= 8:
            return 8
        return 0

    def _pp_prefill_maybe_log_age(
        self: Scheduler,
        tag: str,
        req: Optional[Req],
        field: str,
        waiting_queue,
        bootstrap_queue,
        inflight_queue,
    ) -> None:
        if req is None:
            return
        age_sec = self._pp_prefill_req_age_sec(req, field)
        bucket = self._pp_prefill_age_bucket(age_sec)
        if bucket == 0:
            return
        self._pp_prefill_problem_log(
            tag,
            key=(req.rid, bucket),
            rid=req.rid,
            age_sec=round(age_sec, 1),
            bootstrap_room=getattr(req, "bootstrap_room", None),
            batch=[],
            waiting=self._pp_prefill_queue_shape(waiting_queue),
            bootstrap=self._pp_prefill_queue_shape(bootstrap_queue),
            inflight=self._pp_prefill_diag_rids(inflight_queue),
            prealloc=len(getattr(self, "disagg_prefill_prealloc_queue", [])),
        )

    def _pp_build_req_payload(self: Scheduler, recv_reqs):
        if self.pp_group.is_last_rank:
            return recv_reqs
        events = []
        if (
            self.enable_hicache_storage
            and self.tree_cache is not None
            and hasattr(self.tree_cache, "consume_pp_host_tree_events")
        ):
            events = self.tree_cache.consume_pp_host_tree_events()
        return self._pp_pack_req_payload(recv_reqs, events)

    def _pp_apply_hicache_sync_before_batch(self: Scheduler) -> None:
        if (
            self.pp_rank == 0
            or not self.enable_hicache_storage
            or self.tree_cache is None
        ):
            return
        incoming_events = getattr(self, "pp_hicache_host_tree_events", None) or []
        if incoming_events and hasattr(self.tree_cache, "enqueue_pp_host_tree_events"):
            self.tree_cache.enqueue_pp_host_tree_events(incoming_events)
        self.pp_hicache_host_tree_events = []
        # Keep PP transport minimal: only stage/broadcast the ordered host-tree events
        # here. The actual local application should stay on the original hicache pump
        # paths (writing_check/check_prefetch_progress), otherwise batch selection can
        # interleave with drain_storage_control_queues() and disturb PD bootstrap timing.

    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.recv_requests()
                    self.process_input_requests(recv_reqs)
                with torch.profiler.record_function("get_next_batch_to_run"):
                    self._pp_apply_hicache_sync_before_batch()
                    self.mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work, kind="proxy")
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work, kind="req")
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            self._pp_build_req_payload(recv_reqs),
                            async_send=True,
                        )
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.self_check_during_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []
        prefill_diag_enabled = self._pp_prefill_diag_enabled()

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                self._pp_apply_hicache_sync_before_batch()
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work, kind="req")
                if prefill_diag_enabled:
                    recv_diag = self._pp_prefill_diag_rids(recv_reqs)
                    waiting_diag = self._pp_prefill_diag_rids(self.waiting_queue)
                    bootstrap_diag = self._pp_prefill_diag_rids(
                        self.disagg_prefill_bootstrap_queue.queue
                    )
                    inflight_diag = self._pp_prefill_diag_rids(
                        self.disagg_prefill_inflight_queue
                    )
                    if recv_diag or waiting_diag or bootstrap_diag or inflight_diag:
                        self._pp_prefill_diag_log(
                            "recv",
                            mb=mb_id,
                            recv=recv_diag,
                            waiting=waiting_diag,
                            bootstrap_q=bootstrap_diag,
                            inflight_q=inflight_diag,
                        )
                    self._pp_prefill_maybe_log_age(
                        "bootstrap_head_age",
                        self.disagg_prefill_bootstrap_queue.queue[0]
                        if self.disagg_prefill_bootstrap_queue.queue
                        else None,
                        "prefill_bootstrap_queue_entry_time",
                        waiting_queue=self.waiting_queue,
                        bootstrap_queue=self.disagg_prefill_bootstrap_queue.queue,
                        inflight_queue=self.disagg_prefill_inflight_queue,
                    )
                    self._pp_prefill_maybe_log_age(
                        "inflight_head_age",
                        self.disagg_prefill_inflight_queue[0]
                        if self.disagg_prefill_inflight_queue
                        else None,
                        "prefill_transfer_queue_entry_time",
                        waiting_queue=self.waiting_queue,
                        bootstrap_queue=self.disagg_prefill_bootstrap_queue.queue,
                        inflight_queue=self.disagg_prefill_inflight_queue,
                    )

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids(mb_id)
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(
                    send_bootstrapped_work, kind="bootstrap"
                )
                if prefill_diag_enabled and (bootstrapped_rids[0] or bootstrapped_rids[1]):
                    self._pp_prefill_diag_log(
                        "bootstrap_poll",
                        mb=mb_id,
                        boot_good=self._pp_prefill_diag_rids(bootstrapped_rids[0]),
                        boot_bad=self._pp_prefill_diag_rids(bootstrapped_rids[1]),
                    )
                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work, kind="transfer")
                tmbs[mb_id] = transferred_rids
                if prefill_diag_enabled and transferred_rids:
                    self._pp_prefill_diag_log(
                        "transfer_poll",
                        mb=mb_id,
                        transferred=self._pp_prefill_diag_rids(transferred_rids),
                    )

                self.process_prefill_chunk()
                batch = self.get_new_batch_prefill()
                batch = self.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch
                if prefill_diag_enabled and (batch or self.waiting_queue):
                    self._pp_prefill_diag_log(
                        "batch_pick",
                        mb=mb_id,
                        batch=self._pp_prefill_diag_rids(batch.reqs if batch else []),
                        waiting_after_pick=self._pp_prefill_diag_rids(self.waiting_queue),
                    )
                if (
                    prefill_diag_enabled
                    and (batch or self.waiting_queue or self.disagg_prefill_bootstrap_queue.queue)
                ):
                    batch_rids = tuple(
                        self._pp_prefill_diag_rids(batch.reqs if batch else [])
                    )
                    waiting_rids = tuple(self._pp_prefill_diag_rids(self.waiting_queue))
                    bootstrap_rids = tuple(
                        self._pp_prefill_diag_rids(
                            self.disagg_prefill_bootstrap_queue.queue
                        )
                    )
                    inflight_rids = tuple(
                        self._pp_prefill_diag_rids(self.disagg_prefill_inflight_queue)
                    )
                    self._pp_prefill_shape_snapshot_log(
                        "batch_shape_snapshot",
                        key=(mb_id, batch_rids, waiting_rids, bootstrap_rids, inflight_rids),
                        batch=batch,
                        waiting_queue=self.waiting_queue,
                        bootstrap_queue=self.disagg_prefill_bootstrap_queue.queue,
                        inflight_queue=self.disagg_prefill_inflight_queue,
                    )

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work, kind="proxy")
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs,
                        next_first_rank_mb_id,
                        release_rids,
                        transferred_rids,
                    )
                )

                if bmbs[next_mb_id] is not None:
                    recv_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    recv_good, recv_bad, recv_deferred, _recv_shared_capacity = (
                        self._pp_unpack_bootstrap_payload(
                            recv_consensus_bootstrapped_rids
                        )
                    )
                    if prefill_diag_enabled and (recv_good or recv_bad or recv_deferred):
                        self._pp_prefill_diag_log(
                            "bootstrap_consensus_recv",
                            mb=next_mb_id,
                            consensus_good=self._pp_prefill_diag_rids(recv_good),
                            consensus_bad=self._pp_prefill_diag_rids(recv_bad),
                            consensus_deferred=self._pp_prefill_diag_rids(
                                recv_deferred
                            ),
                        )
                    if self.pp_group.is_last_rank:
                        next_consensus_bootstrapped_rids = bmbs[next_mb_id]
                        local_good, local_bad, local_deferred, _local_shared_capacity = (
                            self._pp_unpack_bootstrap_payload(
                                next_consensus_bootstrapped_rids
                            )
                        )
                        if prefill_diag_enabled and (local_good or local_bad or local_deferred):
                            self._pp_prefill_diag_log(
                                "bootstrap_apply_source",
                                mb=next_mb_id,
                                source="local_consensus",
                                local_good=self._pp_prefill_diag_rids(local_good),
                                local_bad=self._pp_prefill_diag_rids(local_bad),
                                local_deferred=self._pp_prefill_diag_rids(
                                    local_deferred
                                ),
                                recv_good=self._pp_prefill_diag_rids(recv_good),
                                recv_bad=self._pp_prefill_diag_rids(recv_bad),
                                recv_deferred=self._pp_prefill_diag_rids(
                                    recv_deferred
                                ),
                            )
                    else:
                        next_consensus_bootstrapped_rids = (
                            recv_consensus_bootstrapped_rids
                        )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                    released_good = self._pp_prefill_diag_rids(
                        next_consensus_bootstrapped_rids[0]
                        if next_consensus_bootstrapped_rids
                        else []
                    )
                    released_bad = self._pp_prefill_diag_rids(
                        next_consensus_bootstrapped_rids[1]
                        if next_consensus_bootstrapped_rids
                        else []
                    )
                    released_deferred = self._pp_prefill_diag_rids(
                        next_consensus_bootstrapped_rids[2]
                        if next_consensus_bootstrapped_rids
                        and len(next_consensus_bootstrapped_rids) > 3
                        else []
                    )
                    waiting_after_apply = self._pp_prefill_diag_rids(self.waiting_queue)
                    if prefill_diag_enabled and (
                        released_good
                        or released_bad
                        or released_deferred
                        or waiting_after_apply
                    ):
                        self._pp_prefill_diag_log(
                            "bootstrap_apply",
                            mb=next_mb_id,
                            released_good=released_good,
                            released_bad=released_bad,
                            released_deferred=released_deferred,
                            waiting_after_apply=waiting_after_apply,
                        )
                    if prefill_diag_enabled and released_good:
                        for rid in released_good[:8]:
                            logger.warning(
                                "[PPReqPhase] pp=%s cp=%s tp=%s rid=%s phase=bootstrap_to_waiting mb=%s",
                                self.pp_rank,
                                self.attn_cp_rank,
                                self.attn_tp_rank,
                                rid,
                                next_mb_id,
                            )
                    # Consume this microbatch's bootstrap consensus exactly once.
                    bmbs[next_mb_id] = None
                self._pp_commit_comm_work(
                    send_consensus_bootstrapped_work, kind="consensus_bootstrap"
                )
                if tmbs[next_mb_id] is not None:
                    next_release_payload = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids, _ack_mb_id, _ack_rids, _ack_barrier_rid = (
                        self._pp_unpack_release_payload(next_release_payload)
                    )
                    if prefill_diag_enabled and next_release_rids:
                        self._pp_prefill_diag_log(
                            "release_recv",
                            mb=next_mb_id,
                            release=self._pp_prefill_diag_rids(next_release_rids),
                        )
                self._pp_commit_comm_work(send_release_work, kind="release")
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                    inflight_after_apply = self._pp_prefill_diag_rids(
                        self.disagg_prefill_inflight_queue
                    )
                    waiting_after_release = self._pp_prefill_diag_rids(
                        self.waiting_queue
                    )
                    if prefill_diag_enabled and (
                        next_release_rids or inflight_after_apply or waiting_after_release
                    ):
                        self._pp_prefill_diag_log(
                            "release_apply",
                            mb=next_mb_id,
                            inflight_after_apply=inflight_after_apply,
                            waiting_after_release=waiting_after_release,
                        )
                    if prefill_diag_enabled:
                        self._pp_prefill_maybe_log_age(
                            "inflight_head_age",
                            self.disagg_prefill_inflight_queue[0]
                            if self.disagg_prefill_inflight_queue
                            else None,
                            "prefill_transfer_queue_entry_time",
                            waiting_queue=self.waiting_queue,
                            bootstrap_queue=self.disagg_prefill_bootstrap_queue.queue,
                            inflight_queue=self.disagg_prefill_inflight_queue,
                        )
                    # Consume this microbatch's release consensus exactly once.
                    tmbs[next_mb_id] = None
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        self._pp_build_req_payload(recv_reqs), async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.self_check_during_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work, kind="req")

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work, kind="transfer")

                # get batch to run and proxy tensors if needed
                batch = self.get_next_disagg_decode_batch_to_run()
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not self.cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work, kind="proxy")

                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work, kind="release")

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch and not self.cur_batch.forward_mode.is_prebuilt():
                        torch.cuda.current_stream().wait_event(self.launch_event)
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.self_check_during_idle()

    def init_pp_loop_state(self: Scheduler):
        self.pp_loop_size: int = self.pp_size + self.server_args.pp_async_batch_depth
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        self.require_attn_tp_allgather = (
            not self.server_args.enable_nsa_prefill_context_parallel
        )
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]] = (
            deque()
        )

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None
        self._pp_tensor_dict_inbox: Dict[str, deque[Dict[str, torch.Tensor]]] = (
            defaultdict(deque)
        )
        self.pp_launch_frontier_ack_by_mb: Dict[int, Dict[str, object]] = {}
        self.pp_pending_launch_frontier_ack_by_mb: Dict[int, Dict[str, object]] = {}
        self.pp_current_prefill_mb_id: Optional[int] = None

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list = []
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = np.random.randint(
                    0, 10000, size=chunk_size, dtype=np.int64
                ).tolist()
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.logprob_start_len = -1
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = len(req.fill_ids)

                if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                }

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with CUDA synchronization for accurate timing
                # Synchronize before starting timing to ensure clean measurement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                batch.prepare_for_extend()
                model_worker_batch = batch.get_model_worker_batch()

                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(req.fill_ids)
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    self.req_to_token_pool.free(req)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

            if self.attn_cp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = getattr(self, "max_prefill_tokens", None)
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # Apply the consensus snapshot without re-polling local sender state.
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
                deferred_consensus_bootstrapped_rids,
                shared_bootstrap_capacity,
            ) = self._pp_unpack_bootstrap_payload(bootstrapped_rids)
            deferred_reqs = self._pp_prefill_defer_bootstrap_rids(
                deferred_consensus_bootstrapped_rids
            )
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.apply_bootstrap_consensus(
                    good_consensus_bootstrapped_rids,
                    bad_consensus_bootstrapped_rids,
                    return_failed_reqs=True,
                )
            )
            if hasattr(self.tree_cache, "discard_pp_locally_revoked_req"):
                for req in failed_reqs:
                    self.tree_cache.discard_pp_locally_revoked_req(req.rid)
            elif hasattr(self.tree_cache, "pp_locally_revoked_req_ids"):
                for req in failed_reqs:
                    self.tree_cache.pp_locally_revoked_req_ids.discard(req.rid)
            self.waiting_queue.extend(good_reqs)
            if (
                good_reqs
                and self.enable_hicache_storage
                and self.pp_group is not None
                and not self.pp_group.is_first_rank
            ):
                # Do not let fresh bootstrap arrivals perturb the current pick on
                # follow ranks, but once bootstrap consensus admits them into the
                # waiting queue we still need to warm their storage state before
                # they become waiting-head candidates.
                for req in good_reqs:
                    self._prefetch_kvcache(req)
            if self._pp_prefill_diag_enabled() and (
                good_consensus_bootstrapped_rids
                or bad_consensus_bootstrapped_rids
                or good_reqs
                or failed_reqs
                or self.waiting_queue
                or self.disagg_prefill_bootstrap_queue.queue
            ):
                logger.warning(
                    "[PPPrefillDiag][bootstrap_apply] pp=%s cp=%s tp=%s "
                    "consensus_good=%s consensus_bad=%s consensus_deferred=%s "
                    "popped_good=%s popped_failed=%s popped_deferred=%s "
                    "shared_capacity=%s waiting=%s bootstrap=%s waiting_head=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    self.attn_tp_rank,
                    good_consensus_bootstrapped_rids,
                    bad_consensus_bootstrapped_rids,
                    deferred_consensus_bootstrapped_rids,
                    [req.rid for req in good_reqs],
                    [req.rid for req in failed_reqs],
                    deferred_reqs,
                    shared_bootstrap_capacity,
                    len(self.waiting_queue),
                    len(self.disagg_prefill_bootstrap_queue.queue),
                    self._pp_prefill_diag_queue(list(self.waiting_queue)),
                )
            return [
                [req.rid for req in good_reqs],
                [req.rid for req in failed_reqs],
                deferred_reqs,
                shared_bootstrap_capacity,
            ]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus bootstrapp reqs
        local_bootstrap_capacity = (
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.available_size()
        )
        deferred_bootstrapped_rids = []
        for req in self.disagg_prefill_bootstrap_queue.queue:
            if req.bootstrap_mb_id is None:
                req.bootstrap_mb_id = mb_id
        local_bootstrap_reqs = [
            req
            for req in self.disagg_prefill_bootstrap_queue.queue
            if req.bootstrap_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            (
                good_bootstrapped_rids,
                bad_bootstrapped_rids,
                _local_candidate_polls,
            ) = self.disagg_prefill_bootstrap_queue.get_bootstrapped_rids(
                local_bootstrap_reqs, return_polls=True
            )
            shared_bootstrap_capacity = local_bootstrap_capacity
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            (
                prev_good_bootstrapped_rids,
                prev_bad_bootstrapped_rids,
                _prev_deferred_bootstrapped_rids,
                prev_shared_bootstrap_capacity,
            ) = self._pp_unpack_bootstrap_payload(prev_bootstrapped_rids)
            prev_good_rids_set = set(prev_good_bootstrapped_rids)
            local_candidate_reqs = [
                req
                for req in local_bootstrap_reqs
                if req.rid in prev_good_rids_set
            ]
            rebound_rids = []
            if prev_good_bootstrapped_rids and not local_candidate_reqs:
                # PP0 has already proposed a non-empty bootstrap frontier. If PP1
                # filtered everything away only because requests were latched to the
                # other microbatch, rebind those matching upstream candidates to the
                # current mb and retry locally instead of stalling forever.
                for req in self.disagg_prefill_bootstrap_queue.queue:
                    if req.rid in prev_good_rids_set and req.bootstrap_mb_id != mb_id:
                        req.bootstrap_mb_id = mb_id
                        rebound_rids.append(req.rid)
                if rebound_rids:
                    local_bootstrap_reqs = [
                        req
                        for req in self.disagg_prefill_bootstrap_queue.queue
                        if req.bootstrap_mb_id == mb_id
                    ]
                    local_candidate_reqs = [
                        req
                        for req in local_bootstrap_reqs
                        if req.rid in prev_good_rids_set
                    ]
            (
                curr_good_bootstrapped_rids,
                curr_bad_bootstrapped_rids,
                candidate_polls,
            ) = (
                self.disagg_prefill_bootstrap_queue.get_bootstrapped_rids(
                    local_candidate_reqs, return_polls=True
                )
            )
            local_bootstrap_rids = [req.rid for req in local_bootstrap_reqs]
            head_intermediate_rid = None
            head_intermediate_count = 0
            if (
                prev_good_bootstrapped_rids
                and local_candidate_reqs
                and not curr_good_bootstrapped_rids
                and not curr_bad_bootstrapped_rids
            ):
                head_candidate = local_candidate_reqs[0]
                revoked_head_rid = (
                    self.tree_cache.peek_pp_locally_revoked_req()
                    if hasattr(self.tree_cache, "peek_pp_locally_revoked_req")
                    else None
                )
                locally_revoked = head_candidate.rid == revoked_head_rid
                if (
                    local_bootstrap_reqs
                    and local_bootstrap_reqs[0].rid == head_candidate.rid
                    and prev_good_bootstrapped_rids[0] == head_candidate.rid
                ):
                    head_intermediate_rid = head_candidate.rid
                    if locally_revoked:
                        deferred_bootstrapped_rids = [head_intermediate_rid]
                        (
                            curr_good_bootstrapped_rids,
                            curr_bad_bootstrapped_rids,
                            candidate_polls,
                        ) = self.disagg_prefill_bootstrap_queue.get_bootstrapped_rids(
                            local_candidate_reqs[1:], return_polls=True
                        )
                    else:
                        head_intermediate_count = self._pp_prefill_intermediate_head_count(
                            head_intermediate_rid
                        )
                        if head_intermediate_count >= 32:
                            deferred_bootstrapped_rids = [head_intermediate_rid]
                            (
                                curr_good_bootstrapped_rids,
                                curr_bad_bootstrapped_rids,
                                candidate_polls,
                            ) = self.disagg_prefill_bootstrap_queue.get_bootstrapped_rids(
                                local_candidate_reqs[1:], return_polls=True
                            )
                else:
                    self._pp_prefill_clear_intermediate_head(
                        local_bootstrap_reqs[0].rid if local_bootstrap_reqs else None
                    )
            elif local_bootstrap_reqs:
                self._pp_prefill_clear_intermediate_head(local_bootstrap_reqs[0].rid)
            effective_prev_good_bootstrapped_rids = list(prev_good_bootstrapped_rids)
            if (
                deferred_bootstrapped_rids
                and effective_prev_good_bootstrapped_rids
                and effective_prev_good_bootstrapped_rids[0]
                == deferred_bootstrapped_rids[0]
            ):
                effective_prev_good_bootstrapped_rids = (
                    effective_prev_good_bootstrapped_rids[len(deferred_bootstrapped_rids) :]
                )
            good_bootstrapped_rids = _ordered_common_prefix(
                effective_prev_good_bootstrapped_rids, curr_good_bootstrapped_rids
            )
            # Treat upstream abort/fail as authoritative, but only consume the
            # contiguous local queue prefix to preserve FIFO semantics.
            effective_prev_bad_bootstrapped_rids = list(prev_bad_bootstrapped_rids)
            effective_local_bootstrap_rids = list(local_bootstrap_rids)
            if deferred_bootstrapped_rids:
                effective_local_bootstrap_rids = effective_local_bootstrap_rids[
                    len(deferred_bootstrapped_rids) :
                ]
            bad_bootstrapped_rids = _ordered_prefix_from_queue(
                effective_local_bootstrap_rids, effective_prev_bad_bootstrapped_rids
            )
            shared_bootstrap_capacity = min(
                prev_shared_bootstrap_capacity, local_bootstrap_capacity
            )
            if prev_good_bootstrapped_rids and not curr_good_bootstrapped_rids:
                self._pp_prefill_problem_log(
                    "bootstrap_no_local_candidate",
                    key=(
                        tuple(prev_good_bootstrapped_rids[:4]),
                        tuple(local_bootstrap_rids[:4]),
                    ),
                    prev_good=prev_good_bootstrapped_rids[:8],
                    local_bootstrap=local_bootstrap_rids[:8],
                    curr_bad=curr_bad_bootstrapped_rids[:8],
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if rebound_rids:
                self._pp_prefill_problem_log(
                    "bootstrap_mb_rebind",
                    key=(mb_id, tuple(rebound_rids[:4])),
                    prev_good=prev_good_bootstrapped_rids[:8],
                    rebound=rebound_rids[:8],
                    poll=self._pp_prefill_bootstrap_poll_summary(
                        mb_id,
                        local_bootstrap_reqs,
                        local_candidate_reqs,
                        curr_good_bootstrapped_rids,
                        curr_bad_bootstrapped_rids,
                        candidate_polls,
                    ),
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if (
                prev_good_bootstrapped_rids
                and not curr_good_bootstrapped_rids
                and not curr_bad_bootstrapped_rids
            ):
                self._pp_prefill_problem_log(
                    "bootstrap_poll_empty",
                    key=(
                        mb_id,
                        tuple(prev_good_bootstrapped_rids[:4]),
                        tuple(local_bootstrap_rids[:4]),
                    ),
                    prev_good=prev_good_bootstrapped_rids[:8],
                    poll=self._pp_prefill_bootstrap_poll_summary(
                        mb_id,
                        local_bootstrap_reqs,
                        local_candidate_reqs,
                        curr_good_bootstrapped_rids,
                        curr_bad_bootstrapped_rids,
                        candidate_polls,
                    ),
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if head_intermediate_rid is not None:
                self._pp_prefill_problem_log(
                    "bootstrap_head_intermediate",
                    key=(mb_id, head_intermediate_rid, min(head_intermediate_count, 32)),
                    rid=head_intermediate_rid,
                    count=head_intermediate_count,
                    deferred=bool(
                        deferred_bootstrapped_rids
                        and deferred_bootstrapped_rids[0] == head_intermediate_rid
                    ),
                    poll=self._pp_prefill_bootstrap_poll_summary(
                        mb_id,
                        local_bootstrap_reqs,
                        local_candidate_reqs,
                        curr_good_bootstrapped_rids,
                        curr_bad_bootstrapped_rids,
                        candidate_polls,
                    ),
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            elif (
                prev_good_bootstrapped_rids
                and not curr_good_bootstrapped_rids
            ):
                self._pp_prefill_problem_log(
                    "bootstrap_poll_sparse",
                    key=(
                        mb_id,
                        tuple(prev_good_bootstrapped_rids[:4]),
                        tuple(curr_bad_bootstrapped_rids[:4]),
                    ),
                    prev_good=prev_good_bootstrapped_rids[:8],
                    poll=self._pp_prefill_bootstrap_poll_summary(
                        mb_id,
                        local_bootstrap_reqs,
                        local_candidate_reqs,
                        curr_good_bootstrapped_rids,
                        curr_bad_bootstrapped_rids,
                        candidate_polls,
                    ),
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if (
                prev_good_bootstrapped_rids
                and curr_good_bootstrapped_rids
                and not good_bootstrapped_rids
            ):
                self._pp_prefill_problem_log(
                    "bootstrap_prefix_collapsed",
                    key=(
                        tuple(prev_good_bootstrapped_rids[:4]),
                        tuple(curr_good_bootstrapped_rids[:4]),
                    ),
                    prev_good=prev_good_bootstrapped_rids[:8],
                    curr_good=curr_good_bootstrapped_rids[:8],
                    local_bootstrap=local_bootstrap_rids[:8],
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if prev_bad_bootstrapped_rids and not bad_bootstrapped_rids:
                self._pp_prefill_problem_log(
                    "abort_cleanup_drift",
                    key=(
                        tuple(prev_bad_bootstrapped_rids[:4]),
                        tuple(local_bootstrap_rids[:4]),
                    ),
                    prev_bad=prev_bad_bootstrapped_rids[:8],
                    local_bootstrap=local_bootstrap_rids[:8],
                    curr_bad=curr_bad_bootstrapped_rids[:8],
                    bootstrap=len(self.disagg_prefill_bootstrap_queue.queue),
                )
            if self._pp_prefill_diag_enabled() and (
                prev_good_bootstrapped_rids
                or curr_good_bootstrapped_rids
                or prev_bad_bootstrapped_rids
                or curr_bad_bootstrapped_rids
                or deferred_bootstrapped_rids
            ):
                logger.warning(
                    "[PPPrefillDiag][bootstrap_intersection] pp=%s cp=%s tp=%s "
                    "prev_good=%s curr_good=%s merged_good=%s prev_bad=%s curr_bad=%s "
                    "merged_bad=%s deferred=%s shared_capacity=%s local_capacity=%s bootstrap=%s "
                    "poll=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    self.attn_tp_rank,
                    prev_good_bootstrapped_rids,
                    curr_good_bootstrapped_rids,
                    good_bootstrapped_rids,
                    prev_bad_bootstrapped_rids,
                    curr_bad_bootstrapped_rids,
                    bad_bootstrapped_rids,
                    deferred_bootstrapped_rids,
                    shared_bootstrap_capacity,
                    local_bootstrap_capacity,
                    len(self.disagg_prefill_bootstrap_queue.queue),
                    self._pp_prefill_bootstrap_poll_summary(
                        mb_id,
                        local_bootstrap_reqs,
                        local_candidate_reqs,
                        curr_good_bootstrapped_rids,
                        curr_bad_bootstrapped_rids,
                        candidate_polls,
                    ),
                )
        if len(good_bootstrapped_rids) > shared_bootstrap_capacity:
            good_bootstrapped_rids = good_bootstrapped_rids[:shared_bootstrap_capacity]
        if (
            self._pp_prefill_diag_enabled()
            and self.pp_group.is_first_rank
            and (good_bootstrapped_rids or bad_bootstrapped_rids or deferred_bootstrapped_rids)
        ):
            logger.warning(
                "[PPPrefillDiag][bootstrap_poll] pp=%s cp=%s tp=%s local_good=%s "
                "local_bad=%s local_deferred=%s shared_capacity=%s bootstrap=%s",
                self.pp_rank,
                self.attn_cp_rank,
                self.attn_tp_rank,
                good_bootstrapped_rids,
                bad_bootstrapped_rids,
                deferred_bootstrapped_rids,
                shared_bootstrap_capacity,
                len(self.disagg_prefill_bootstrap_queue.queue),
            )
        return [
            good_bootstrapped_rids,
            bad_bootstrapped_rids,
            deferred_bootstrapped_rids,
            shared_bootstrap_capacity,
        ]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = _ordered_intersection(
                prev_transferred_rids, curr_transferred_rids
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            # Keep the send phase aligned with the local apply phase on last rank.
            # `consensus_bootstrapped_rids` is populated from the previously applied slot
            # at the end of the loop, so forwarding it here keeps PP0 and PP1 on the
            # same admitted bootstrap snapshot.
            if bmbs[next_first_rank_mb_id] is not None:
                if consensus_bootstrapped_rids is None:
                    # Seed the delayed-send pipeline with a well-formed empty snapshot.
                    # This preserves the recv/send cadence without letting PP0 consume
                    # current-slot bootstrap consensus ahead of the last rank's local apply.
                    consensus_bootstrapped_rids = [[], [], [], 0]
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
        launch_ack_mb_id: Optional[int] = None,
        launch_ack_rids: Optional[List[str]] = None,
        launch_ack_barrier_rid: Optional[str] = None,
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = self._pp_pack_release_payload(
                    transferred_rids,
                    launch_ack_mb_id=launch_ack_mb_id,
                    launch_ack_rids=launch_ack_rids,
                    launch_ack_barrier_rid=launch_ack_barrier_rid,
                )
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(
        self: Scheduler, work: List[P2PWork], kind: Optional[str] = None
    ) -> None:
        start = time.perf_counter()
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > 0 and hasattr(self, "_record_pp_comm_wait_timing"):
            self._record_pp_comm_wait_timing(elapsed_ms, kind=kind)

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[PPProxyTensors, GenerationBatchResult, torch.cuda.Event]:
        self._pp_commit_comm_work(work=self.send_output_work, kind="output")
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        start = time.perf_counter()
        p2p_work = []
        if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.pp_rank * self.tp_size + dp_offset,
                ((self.pp_rank + 1) % self.pp_size) * self.tp_size + dp_offset,
                async_send=async_send,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if hasattr(self, "_record_pp_comm_send_timing"):
            self._record_pp_comm_send_timing(elapsed_ms)
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        recv_start = time.perf_counter()
        if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.pp_rank - 1) % self.pp_size) * self.tp_size + dp_offset,
                self.pp_rank * self.tp_size + dp_offset,
            )
        else:
            data = None
        if hasattr(self, "_record_pp_comm_recv_timing"):
            self._record_pp_comm_recv_timing((time.perf_counter() - recv_start) * 1000.0)

        if self.attn_tp_size > 1:
            tp_bcast_start = time.perf_counter()
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )
            if hasattr(self, "_record_pp_comm_tp_bcast_timing"):
                self._record_pp_comm_tp_bcast_timing(
                    (time.perf_counter() - tp_bcast_start) * 1000.0
                )

        if self.attn_cp_size > 1:
            cp_bcast_start = time.perf_counter()
            data = broadcast_pyobj(
                data,
                self.attn_cp_group.rank,
                self.attn_cp_cpu_group,
                src=self.attn_cp_group.ranks[0],
            )
            if hasattr(self, "_record_pp_comm_cp_bcast_timing"):
                self._record_pp_comm_cp_bcast_timing(
                    (time.perf_counter() - cp_bcast_start) * 1000.0
                )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
    ):
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Dict[str, torch.Tensor]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.
        """
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            tensor_dict = self.pp_group.recv_tensor_dict(
                all_gather_group=all_gather_group
            )
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict
            else:
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(tensor_dict)

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self._pp_recv_typed_dict(
                    expected_kind="proxy",
                    all_gather_group=(
                        self.attn_tp_group if self.require_attn_tp_allgather else None
                    ),
                )
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        batch.output_ids = pp_outputs["next_token_ids"]
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            if mbs[next_first_rank_mb_id] is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if not mbs[next_first_rank_mb_id].forward_mode.is_prebuilt():
                    torch.cuda.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[PPProxyTensors, List[P2PWork], torch.cuda.Event]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = self._pp_send_output_to_next_stage(
            next_first_rank_mb_id,
            mbs,
            last_rank_comm_queue,
            pp_outputs,
        )

        if mbs[next_mb_id] is not None:
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = None
                if not mbs[next_mb_id].forward_mode.is_prebuilt():
                    next_pp_outputs = PPProxyTensors(
                        self._pp_recv_dict_from_prev_stage()
                    )
            if not mbs[next_mb_id].forward_mode.is_prebuilt():
                with self.copy_stream_ctx:
                    self.copy_stream.wait_stream(self.schedule_stream)
                    batch_result = self._pp_prep_batch_result(
                        mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
                    )
                    d2h_event = torch.cuda.Event()
                    d2h_event.record(torch.cuda.current_stream())

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                result = self.run_batch(self.cur_batch, pp_proxy_tensors)
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, self.cur_batch)
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return _ordered_intersection(prev_retract_rids, curr_retract_rids)

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = _ordered_intersection(
                prev_good_prealloc_rids, curr_good_prealloc_rids
            )
            bad_prealloc_rids = _ordered_union(
                prev_bad_prealloc_rids, curr_bad_prealloc_rids
            )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = _ordered_intersection(
                prev_transferred_rids, curr_transferred_rids
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                rids_to_check=good_consensus_prealloc_rids
                + bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
