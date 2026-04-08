from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_nsa_hybrid_stack,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_PP_EVENT_WRITE_BACKUP_COMMITTED = 0
_PP_EVENT_PREFETCH_SKIP = 1
_PP_EVENT_REVOKE = 2
_PP_EVENT_PREFETCH_FINALIZE = 3

_PP_EVENT_KIND_TO_CODE = {
    "WRITE_BACKUP_COMMITTED": _PP_EVENT_WRITE_BACKUP_COMMITTED,
    "PREFETCH_SKIP": _PP_EVENT_PREFETCH_SKIP,
    "REVOKE": _PP_EVENT_REVOKE,
    "PREFETCH_FINALIZE": _PP_EVENT_PREFETCH_FINALIZE,
}
_PP_EVENT_CODE_TO_KIND = {value: key for key, value in _PP_EVENT_KIND_TO_CODE.items()}
_PP_HOST_TREE_EVENTS_V1 = "__pp_host_tree_events_v1__"
_PP_LIGHT_EVENT_HEADER = struct.Struct("<BQII")


class _HiCacheDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if os.getenv("SGLANG_DEBUG_HICACHE_VERBOSE", "0") == "1":
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not (
            message.startswith("[HiCache")
            or message.startswith("[PPReqPhase]")
            or message.startswith("[PPHiCacheSync]")
        )


if not any(isinstance(f, _HiCacheDebugFilter) for f in logger.filters):
    logger.addFilter(_HiCacheDebugFilter())


def _safe_attn_tp_rank(obj) -> int:
    return int(getattr(obj, "attn_tp_rank", 0))


@dataclass
class PPHostTreeEvent:
    seq: int
    kind: str
    rid: Optional[str] = None
    loaded_from_storage: int = 0
    node_ids: List[int] = field(default_factory=list)
    node_key_lens: List[int] = field(default_factory=list)
    node_last_hashes: List[Optional[str]] = field(default_factory=list)
    node_extra_keys: List[Optional[str]] = field(default_factory=list)
    node_parent_key_lens: List[int] = field(default_factory=list)
    node_parent_last_hashes: List[Optional[str]] = field(default_factory=list)
    node_parent_extra_keys: List[Optional[str]] = field(default_factory=list)


def _encode_pp_host_tree_wire_event(event: PPHostTreeEvent):
    kind_code = _PP_EVENT_KIND_TO_CODE[event.kind]
    if kind_code == _PP_EVENT_WRITE_BACKUP_COMMITTED:
        return (
            kind_code,
            event.seq,
            event.rid,
            event.loaded_from_storage,
            tuple(event.node_ids),
            tuple(event.node_key_lens),
            tuple(event.node_last_hashes),
            tuple(event.node_extra_keys),
            tuple(event.node_parent_key_lens),
            tuple(event.node_parent_last_hashes),
            tuple(event.node_parent_extra_keys),
        )
    return (
        kind_code,
        event.seq,
        event.rid,
        event.loaded_from_storage,
    )


def _decode_pp_host_tree_wire_event(event) -> PPHostTreeEvent:
    if isinstance(event, PPHostTreeEvent):
        return event
    if isinstance(event, tuple):
        kind = _PP_EVENT_CODE_TO_KIND[int(event[0])]
        if int(event[0]) == _PP_EVENT_WRITE_BACKUP_COMMITTED:
            return PPHostTreeEvent(
                seq=int(event[1]),
                kind=kind,
                rid=event[2],
                loaded_from_storage=int(event[3]),
                node_ids=[int(v) for v in event[4]],
                node_key_lens=[int(v) for v in event[5]],
                node_last_hashes=list(event[6]),
                node_extra_keys=list(event[7]),
                node_parent_key_lens=[int(v) for v in event[8]],
                node_parent_last_hashes=list(event[9]),
                node_parent_extra_keys=list(event[10]),
            )
        return PPHostTreeEvent(
            seq=int(event[1]),
            kind=kind,
            rid=event[2],
            loaded_from_storage=int(event[3]),
        )
    if isinstance(event, dict):
        return PPHostTreeEvent(
            seq=int(event.get("seq", 0)),
            kind=str(event["kind"]),
            rid=event.get("rid"),
            loaded_from_storage=int(event.get("loaded_from_storage", 0)),
            node_ids=[int(v) for v in event.get("node_ids", [])],
            node_key_lens=[int(v) for v in event.get("node_key_lens", [])],
            node_last_hashes=list(event.get("node_last_hashes", [])),
            node_extra_keys=list(event.get("node_extra_keys", [])),
            node_parent_key_lens=[
                int(v) for v in event.get("node_parent_key_lens", [])
            ],
            node_parent_last_hashes=list(event.get("node_parent_last_hashes", [])),
            node_parent_extra_keys=list(event.get("node_parent_extra_keys", [])),
        )
    raise TypeError(f"Unsupported PP host tree event wire type: {type(event)!r}")


def _encode_pp_light_host_tree_events(events: List[PPHostTreeEvent]) -> bytes:
    out = bytearray()
    for event in events:
        kind_code = _PP_EVENT_KIND_TO_CODE[event.kind]
        if kind_code == _PP_EVENT_WRITE_BACKUP_COMMITTED:
            raise ValueError("write-back events must stay on the heavy path")
        rid_bytes = (event.rid or "").encode("utf-8")
        out.extend(
            _PP_LIGHT_EVENT_HEADER.pack(
                kind_code,
                int(event.seq),
                int(event.loaded_from_storage),
                len(rid_bytes),
            )
        )
        out.extend(rid_bytes)
    return bytes(out)


def _decode_pp_light_host_tree_events(blob: bytes) -> List[PPHostTreeEvent]:
    if not blob:
        return []
    view = memoryview(blob)
    offset = 0
    events: List[PPHostTreeEvent] = []
    while offset < len(view):
        if len(view) < offset + _PP_LIGHT_EVENT_HEADER.size:
            raise ValueError("PP light event blob truncated")
        kind_code, seq, loaded_from_storage, rid_len = _PP_LIGHT_EVENT_HEADER.unpack_from(
            view, offset
        )
        offset += _PP_LIGHT_EVENT_HEADER.size
        end = offset + rid_len
        if len(view) < end:
            raise ValueError("PP light event rid truncated")
        rid = None
        if rid_len:
            rid = view[offset:end].tobytes().decode("utf-8")
        offset = end
        events.append(
            PPHostTreeEvent(
                seq=int(seq),
                kind=_PP_EVENT_CODE_TO_KIND[int(kind_code)],
                rid=rid,
                loaded_from_storage=int(loaded_from_storage),
            )
        )
    return events


def _decode_pp_host_tree_transport(events) -> List[PPHostTreeEvent]:
    if (
        isinstance(events, tuple)
        and len(events) == 3
        and events[0] == _PP_HOST_TREE_EVENTS_V1
    ):
        decoded = _decode_pp_light_host_tree_events(events[1] or b"")
        decoded.extend(
            _decode_pp_host_tree_wire_event(event) for event in (events[2] or ())
        )
        return decoded
    return [_decode_pp_host_tree_wire_event(event) for event in (events or ())]


def _estimate_pp_write_backup_event_bytes(event: PPHostTreeEvent) -> int:
    if event.kind != "WRITE_BACKUP_COMMITTED":
        return 0

    total = 1 + 8 + 4
    total += len((event.rid or "").encode("utf-8"))
    for idx in range(len(event.node_key_lens)):
        total += 8
        total += 4

        last_hash = event.node_last_hashes[idx]
        total += 4 + (len(last_hash.encode("utf-8")) if last_hash is not None else 0)

        extra_key = event.node_extra_keys[idx]
        total += 4 + (len(extra_key.encode("utf-8")) if extra_key is not None else 0)

        if idx < len(event.node_parent_key_lens):
            total += 4
            parent_last_hash = event.node_parent_last_hashes[idx]
            total += 4 + (
                len(parent_last_hash.encode("utf-8"))
                if parent_last_hash is not None
                else 0
            )
            parent_extra_key = event.node_parent_extra_keys[idx]
            total += 4 + (
                len(parent_extra_key.encode("utf-8"))
                if parent_extra_key is not None
                else 0
            )
    return total


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            # Filled by build_nsa_hybrid_stack after storage extra_config is parsed.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(
                "HiRadixCache only supports MHA, MLA, and NSA (DSA) models"
            )

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.attn_cp_rank = params.attn_cp_rank
        self.attn_cp_size = params.attn_cp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, NSATokenToKVPool):
            build_nsa_hybrid_stack(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                attn_cp_rank=self.attn_cp_rank,
                attn_cp_size=self.attn_cp_size,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.prefetch_issue_count_by_reqid: dict[str, int] = {}
        self.zero_hit_prefetch_req_ids: set[str] = set()
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.pp_host_tree_event_seq = 0
        self.pp_outgoing_host_tree_events: List[dict[str, Any]] = []
        self.pp_pending_host_tree_events: Deque[PPHostTreeEvent] = deque()
        self.pp_deferred_revoke_req_ids: Deque[tuple[str, bool]] = deque()
        self.pp_locally_revoked_req_ids: set[str] = set()
        self.pp_locally_revoked_req_queue: Deque[str] = deque()
        self.pp_retry_prefetch_req_ids: set[str] = set()
        self.pp_authoritative_revoked_req_ids: set[str] = set()
        self.pp_soft_skipped_req_ids: set[str] = set()
        self.pp_staged_prefetch_skip_req_ids: set[str] = set()
        self._in_pp_host_tree_replay = False
        self._match_perf_calls = 0
        self._match_perf_total_ms = 0.0
        self._match_perf_max_ms = 0.0
        self._match_perf_total_walk_steps = 0
        self._match_perf_max_walk_steps = 0
        self._match_perf_total_splits = 0
        self._match_perf_max_splits = 0
        self._match_perf_total_host_climb = 0
        self._match_perf_max_host_climb = 0
        self._match_perf_total_backup_climb = 0
        self._match_perf_max_backup_climb = 0
        self._match_perf_total_value_segments = 0
        self._match_perf_total_key_len = 0
        self._match_perf_total_aligned_len = 0
        self._match_perf_total_device_hit = 0
        self._match_perf_total_host_hit = 0
        self._match_perf_root_only_calls = 0
        self._match_perf_split_calls = 0
        self._match_perf_host_climb_calls = 0
        self._match_perf_backup_climb_calls = 0
        self._match_perf_multi_segment_calls = 0
        self._match_perf_device_hit_calls = 0
        self._match_perf_host_hit_calls = 0
        self._match_perf_total_path_fanout = 0
        self._match_perf_max_path_fanout = 0
        self._write_backup_replay_apply_local_ack = 0
        self._write_backup_replay_apply_authoritative = 0
        self._write_backup_replay_miss = 0
        self._tree_counter_last_snapshot = 0
        self._tree_alloc_match_split = 0
        self._tree_alloc_insert_split = 0
        self._tree_alloc_host_insert_split = 0
        self._tree_alloc_device_leaf = 0
        self._tree_alloc_host_leaf = 0
        self._tree_delete_regular_leaf = 0
        self._tree_delete_host_leaf = 0
        self._write_backup_commit_nodes = 0
        self._write_backup_event_count = 0
        self._write_backup_event_total_nodes = 0
        self._write_backup_event_max_nodes = 0
        self._write_backup_event_estimated_bytes = 0
        self._write_backup_event_max_estimated_bytes = 0

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def consume_match_perf_snapshot(self) -> dict[str, int | float]:
        snapshot = {
            "calls": self._match_perf_calls,
            "avg_ms": (
                self._match_perf_total_ms / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_ms": self._match_perf_max_ms,
            "avg_walk": (
                self._match_perf_total_walk_steps / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_walk": self._match_perf_max_walk_steps,
            "avg_splits": (
                self._match_perf_total_splits / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_splits": self._match_perf_max_splits,
            "avg_host_climb": (
                self._match_perf_total_host_climb / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_host_climb": self._match_perf_max_host_climb,
            "avg_backup_climb": (
                self._match_perf_total_backup_climb / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_backup_climb": self._match_perf_max_backup_climb,
            "avg_segments": (
                self._match_perf_total_value_segments / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "avg_key_len": (
                self._match_perf_total_key_len / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "avg_aligned_len": (
                self._match_perf_total_aligned_len / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "avg_device_hit": (
                self._match_perf_total_device_hit / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "avg_host_hit": (
                self._match_perf_total_host_hit / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "root_only_calls": self._match_perf_root_only_calls,
            "split_calls": self._match_perf_split_calls,
            "host_climb_calls": self._match_perf_host_climb_calls,
            "backup_climb_calls": self._match_perf_backup_climb_calls,
            "multi_segment_calls": self._match_perf_multi_segment_calls,
            "device_hit_calls": self._match_perf_device_hit_calls,
            "host_hit_calls": self._match_perf_host_hit_calls,
            "avg_path_fanout": (
                self._match_perf_total_path_fanout / self._match_perf_calls
                if self._match_perf_calls
                else 0.0
            ),
            "max_path_fanout": self._match_perf_max_path_fanout,
        }
        self._match_perf_calls = 0
        self._match_perf_total_ms = 0.0
        self._match_perf_max_ms = 0.0
        self._match_perf_total_walk_steps = 0
        self._match_perf_max_walk_steps = 0
        self._match_perf_total_splits = 0
        self._match_perf_max_splits = 0
        self._match_perf_total_host_climb = 0
        self._match_perf_max_host_climb = 0
        self._match_perf_total_backup_climb = 0
        self._match_perf_max_backup_climb = 0
        self._match_perf_total_value_segments = 0
        self._match_perf_total_key_len = 0
        self._match_perf_total_aligned_len = 0
        self._match_perf_total_device_hit = 0
        self._match_perf_total_host_hit = 0
        self._match_perf_root_only_calls = 0
        self._match_perf_split_calls = 0
        self._match_perf_host_climb_calls = 0
        self._match_perf_backup_climb_calls = 0
        self._match_perf_multi_segment_calls = 0
        self._match_perf_device_hit_calls = 0
        self._match_perf_host_hit_calls = 0
        self._match_perf_total_path_fanout = 0
        self._match_perf_max_path_fanout = 0
        return snapshot

    def consume_write_backup_replay_snapshot(self) -> dict[str, int | float]:
        snapshot = {
            "apply_local_ack": self._write_backup_replay_apply_local_ack,
            "apply_authoritative": self._write_backup_replay_apply_authoritative,
            "miss": self._write_backup_replay_miss,
            "commit_nodes": self._write_backup_commit_nodes,
            "event_count": self._write_backup_event_count,
            "event_avg_nodes": (
                self._write_backup_event_total_nodes / self._write_backup_event_count
                if self._write_backup_event_count
                else 0.0
            ),
            "event_max_nodes": self._write_backup_event_max_nodes,
            "event_est_bytes": self._write_backup_event_estimated_bytes,
            "event_avg_est_bytes": (
                self._write_backup_event_estimated_bytes
                / self._write_backup_event_count
                if self._write_backup_event_count
                else 0.0
            ),
            "event_max_est_bytes": self._write_backup_event_max_estimated_bytes,
            "pending_wb_events": sum(
                1
                for event in self.pp_pending_host_tree_events
                if event.kind == "WRITE_BACKUP_COMMITTED"
            ),
        }
        self._write_backup_replay_apply_local_ack = 0
        self._write_backup_replay_apply_authoritative = 0
        self._write_backup_replay_miss = 0
        self._write_backup_commit_nodes = 0
        self._write_backup_event_count = 0
        self._write_backup_event_total_nodes = 0
        self._write_backup_event_max_nodes = 0
        self._write_backup_event_estimated_bytes = 0
        self._write_backup_event_max_estimated_bytes = 0
        return snapshot

    def get_tree_shape_snapshot(self) -> dict[str, int]:
        stack = [(self.root_node, 0)]
        total_nodes = 0
        evicted_nodes = 0
        backuped_nodes = 0
        leaf_nodes = 0
        max_depth = 0
        max_fanout = 0
        while stack:
            node, depth = stack.pop()
            if node is not self.root_node:
                total_nodes += 1
                if node.evicted:
                    evicted_nodes += 1
                if node.backuped:
                    backuped_nodes += 1
                if not node.children:
                    leaf_nodes += 1
                max_depth = max(max_depth, depth)
            max_fanout = max(max_fanout, len(node.children))
            for child in node.children.values():
                stack.append((child, depth + 1))
        return {
            "nodes": total_nodes,
            "evicted_nodes": evicted_nodes,
            "backuped_nodes": backuped_nodes,
            "leaf_nodes": leaf_nodes,
            "max_depth": max_depth,
            "max_fanout": max_fanout,
            "allocated_node_id": max(0, TreeNode.counter - 1),
        }

    def consume_tree_churn_snapshot(self) -> dict[str, int]:
        current_counter = max(0, TreeNode.counter - 1)
        snapshot = {
            "allocated_delta": max(0, current_counter - self._tree_counter_last_snapshot),
            "alloc_match_split": self._tree_alloc_match_split,
            "alloc_insert_split": self._tree_alloc_insert_split,
            "alloc_host_insert_split": self._tree_alloc_host_insert_split,
            "alloc_device_leaf": self._tree_alloc_device_leaf,
            "alloc_host_leaf": self._tree_alloc_host_leaf,
            "delete_regular_leaf": self._tree_delete_regular_leaf,
            "delete_host_leaf": self._tree_delete_host_leaf,
        }
        self._tree_counter_last_snapshot = current_counter
        self._tree_alloc_match_split = 0
        self._tree_alloc_insert_split = 0
        self._tree_alloc_host_insert_split = 0
        self._tree_alloc_device_leaf = 0
        self._tree_alloc_host_leaf = 0
        self._tree_delete_regular_leaf = 0
        self._tree_delete_host_leaf = 0
        return snapshot

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": self.cache_controller.attn_cp_rank,
                "attn_cp_size": self.cache_controller.attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            if isinstance(self.cache_controller, HybridCacheController):
                self.cache_controller.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=served_model_name,
                    storage_backend_extra_config=extra_config,
                    host_pools=self.cache_controller.mem_pool_host.entries,
                )
            else:
                self.cache_controller.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=served_model_name,
                    storage_backend_extra_config=extra_config,
                )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _normalize_revoke_item(item) -> tuple[str, bool]:
            if isinstance(item, tuple):
                req_id, zero_hit = item
                return req_id, bool(zero_hit)
            return item, False

        def _drain_revoke():
            if self._pp_downstream_sync_enabled():
                for item in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                    req_id, zero_hit = _normalize_revoke_item(item)
                    # Local zero-hit revokes must take effect immediately on this
                    # rank; otherwise the request can remain stuck in
                    # ongoing_prefetch while we wait for an upstream REVOKE that
                    # may never exist (for example when upstream skipped
                    # prefetching entirely).
                    self._drain_single_revoke_req(req_id, zero_hit=zero_hit)
                    self._append_pp_host_tree_event(
                        PPHostTreeEvent(
                            seq=self._next_pp_host_tree_seq(),
                            kind="REVOKE",
                            rid=req_id,
                        )
                    )
                while self.pp_pending_host_tree_events:
                    event = self.pp_pending_host_tree_events[0]
                    if event.kind != "REVOKE" or not self._try_replay_revoke_event(event):
                        break
                    self.pp_pending_host_tree_events.popleft()
                return

            for item in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                req_id, zero_hit = _normalize_revoke_item(item)
                self._drain_single_revoke_req(req_id, zero_hit=zero_hit)
                self._append_pp_host_tree_event(
                    PPHostTreeEvent(
                        seq=self._next_pp_host_tree_seq(),
                        kind="REVOKE",
                        rid=req_id,
                    )
                )

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.prefetch_issue_count_by_reqid.clear()
        self.zero_hit_prefetch_req_ids.clear()
        self.evictable_host_leaves.clear()
        self.pp_outgoing_host_tree_events.clear()
        self.pp_pending_host_tree_events.clear()
        self.pp_deferred_revoke_req_ids.clear()
        self.pp_locally_revoked_req_ids.clear()
        self.pp_locally_revoked_req_queue.clear()
        self.pp_retry_prefetch_req_ids.clear()
        self.pp_authoritative_revoked_req_ids.clear()
        self.pp_soft_skipped_req_ids.clear()
        self.pp_staged_prefetch_skip_req_ids.clear()
        self._tree_counter_last_snapshot = 0
        self._tree_alloc_match_split = 0
        self._tree_alloc_insert_split = 0
        self._tree_alloc_host_insert_split = 0
        self._tree_alloc_device_leaf = 0
        self._tree_alloc_host_leaf = 0
        self._tree_delete_regular_leaf = 0
        self._tree_delete_host_leaf = 0
        self._write_backup_commit_nodes = 0
        super().reset()

    def _pp_downstream_sync_enabled(self) -> bool:
        return self.enable_storage and self.pp_size > 1 and self.pp_rank > 0

    def _pp_write_backup_replay_enabled(self) -> bool:
        return os.getenv("SGLANG_ENABLE_PP_WRITE_BACKUP_REPLAY", "1") == "1"

    def _hicache_verbose_enabled(self) -> bool:
        return os.getenv("SGLANG_DEBUG_HICACHE_VERBOSE", "0") == "1"

    def _pp_should_skip_large_shallow_prefetch(
        self, last_host_node: TreeNode, prefetch_length: int
    ) -> bool:
        # In PP mode, PP0 can run ahead into a new large-suffix window and grow
        # a deep host subtree before PP1 reaches the same request window. That
        # early tree growth is the first observed source of host-hit divergence.
        #
        # Keep this guard intentionally narrow:
        # - PP first rank only
        # - storage-enabled PP only
        # - only large prefetches
        # - only when the current host anchor is still shallow
        if not (self.enable_storage and self.pp_size > 1 and self.pp_rank == 0):
            return False
        if prefetch_length < self.prefetch_threshold:
            return False
        if last_host_node is None or last_host_node.key is None:
            return False
        return len(last_host_node.key) <= 16

    def _append_pp_host_tree_event(self, event: PPHostTreeEvent) -> None:
        if not (self.enable_storage and self.pp_size > 1):
            return
        if (
            event.kind == "WRITE_BACKUP_COMMITTED"
            and not self._pp_write_backup_replay_enabled()
        ):
            return
        # The last PP rank has no downstream peer to consume these events.
        # Keeping them would only create an append-only Python list that grows
        # for the lifetime of the process.
        if self.pp_rank >= self.pp_size - 1:
            return
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPEvent][emit] pp=%s cp=%s seq=%s kind=%s rid=%s loaded=%s outgoing_before=%s",
                self.pp_rank,
                self.attn_cp_rank,
                event.seq,
                event.kind,
                event.rid,
                event.loaded_from_storage,
                len(self.pp_outgoing_host_tree_events),
            )
        self.pp_outgoing_host_tree_events.append(_encode_pp_host_tree_wire_event(event))

    def _next_pp_host_tree_seq(self) -> int:
        self.pp_host_tree_event_seq += 1
        return self.pp_host_tree_event_seq

    def consume_pp_host_tree_events(self) -> Any:
        events = self.pp_outgoing_host_tree_events
        self.pp_outgoing_host_tree_events = []
        if not events:
            return []

        light_events: List[PPHostTreeEvent] = []
        heavy_events = []
        for event in events:
            decoded = _decode_pp_host_tree_wire_event(event)
            if decoded.kind == "WRITE_BACKUP_COMMITTED":
                heavy_events.append(event)
            else:
                light_events.append(decoded)
        transport = (
            _PP_HOST_TREE_EVENTS_V1,
            _encode_pp_light_host_tree_events(light_events),
            tuple(heavy_events),
        )

        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPEvent][consume] pp=%s cp=%s count=%s events=%s",
                self.pp_rank,
                self.attn_cp_rank,
                len(events),
                [
                    (
                        decoded.seq,
                        decoded.kind,
                        decoded.rid,
                        decoded.loaded_from_storage,
                    )
                    for decoded in _decode_pp_host_tree_transport(transport)[:8]
                ],
            )
        return transport

    def prepare_pp_incoming_host_tree_events(
        self, events
    ) -> List[PPHostTreeEvent]:
        if not self._pp_downstream_sync_enabled() or not events:
            return []

        prepared_events = _decode_pp_host_tree_transport(events)
        staged_rids = []
        for decoded in prepared_events:
            if decoded.kind != "PREFETCH_SKIP":
                continue
            req_id = decoded.rid
            if req_id is None:
                continue
            req_id = str(req_id)
            if req_id in self.pp_staged_prefetch_skip_req_ids:
                continue
            self.pp_staged_prefetch_skip_req_ids.add(req_id)
            staged_rids.append(req_id)

        if staged_rids and self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPEvent][stage_prefetch_skip] pp=%s cp=%s count=%s rids=%s",
                self.pp_rank,
                self.attn_cp_rank,
                len(staged_rids),
                staged_rids[:8],
            )

        return prepared_events

    def enqueue_pp_host_tree_events(self, events) -> None:
        if not self._pp_downstream_sync_enabled() or not events:
            return
        decoded_events = _decode_pp_host_tree_transport(events)
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPEvent][enqueue] pp=%s cp=%s count=%s pending_before=%s events=%s",
                self.pp_rank,
                self.attn_cp_rank,
                len(decoded_events),
                len(self.pp_pending_host_tree_events),
                [
                    (
                        decoded.seq,
                        decoded.kind,
                        decoded.rid,
                        decoded.loaded_from_storage,
                    )
                    for decoded in decoded_events[:8]
                ],
            )
        for decoded in decoded_events:
            if (
                decoded.kind == "WRITE_BACKUP_COMMITTED"
                and not self._pp_write_backup_replay_enabled()
            ):
                continue
            self.pp_pending_host_tree_events.append(decoded)

    def stage_pp_incoming_prefetch_skip_events(
        self, events: List[dict[str, Any]]
    ) -> None:
        # Backward-compatible wrapper for older callers.
        self.prepare_pp_incoming_host_tree_events(events)

    def poll_follow_rank_prefetch_issue_action(
        self,
        req_id: str,
        new_input_tokens: List[int],
        prefix_len: int,
        host_hit_length: int,
    ) -> Optional[str]:
        if not self._pp_downstream_sync_enabled():
            return None

        if req_id in self.pp_staged_prefetch_skip_req_ids:
            self.pp_staged_prefetch_skip_req_ids.discard(req_id)
            return "skip"
        return None

    def has_follow_rank_prefetch_issue_pending(self, req_id: str) -> bool:
        return False

    def clear_follow_rank_prefetch_issue_pending(self, req_id: str) -> None:
        self.pp_staged_prefetch_skip_req_ids.discard(req_id)

    def _peek_pp_host_tree_event(self) -> Optional[PPHostTreeEvent]:
        if not self.pp_pending_host_tree_events:
            return None
        return self.pp_pending_host_tree_events[0]

    def has_pending_pp_write_backup_event(self) -> bool:
        if not self._pp_write_backup_replay_enabled():
            return False
        event = self._peek_pp_host_tree_event()
        return event is not None and event.kind == "WRITE_BACKUP_COMMITTED"

    def _collect_node_path_meta(
        self, last_node: Optional[TreeNode]
    ) -> set[tuple[int, Optional[str], Optional[str]]]:
        path_meta: set[tuple[int, Optional[str], Optional[str]]] = set()
        walk_node = last_node
        while walk_node is not None and walk_node is not self.root_node:
            path_meta.add(
                (
                    len(walk_node.key) if walk_node.key is not None else 0,
                    walk_node.get_last_hash_value(),
                    walk_node.key.extra_key if walk_node.key is not None else None,
                )
            )
            walk_node = walk_node.parent
        return path_meta

    def _write_backup_event_affects_req(self, event: PPHostTreeEvent, req) -> bool:
        if event.kind != "WRITE_BACKUP_COMMITTED":
            return False

        path_meta = self._collect_node_path_meta(getattr(req, "last_node", None))
        path_meta.update(
            self._collect_node_path_meta(getattr(req, "last_host_node", None))
        )
        if not path_meta:
            return False

        for idx, key_len in enumerate(event.node_key_lens):
            node_meta = (
                key_len,
                event.node_last_hashes[idx],
                event.node_extra_keys[idx],
            )
            if node_meta in path_meta:
                return True

            if idx < len(event.node_parent_key_lens):
                parent_meta = (
                    event.node_parent_key_lens[idx],
                    event.node_parent_last_hashes[idx],
                    event.node_parent_extra_keys[idx],
                )
                if parent_meta in path_meta:
                    return True
        return False

    def has_pending_pp_write_backup_event_for_req(self, req) -> bool:
        if not self._pp_write_backup_replay_enabled():
            return False
        for event in self.pp_pending_host_tree_events:
            if event.kind != "WRITE_BACKUP_COMMITTED":
                break
            if self._write_backup_event_affects_req(event, req):
                return True
        return False

    def consume_pp_retry_prefetch_req(self, req_id: str) -> bool:
        if req_id not in self.pp_retry_prefetch_req_ids:
            return False
        self.pp_retry_prefetch_req_ids.discard(req_id)
        self.clear_follow_rank_prefetch_issue_pending(req_id)
        # Upstream finalize already decided this request should retry prefetch.
        # Do not keep an older local zero-hit revoke residue blocking waiting-head
        # scheduling for the same req.
        self.discard_pp_locally_revoked_req(req_id)
        self.zero_hit_prefetch_req_ids.discard(req_id)
        return True

    def _pop_pp_host_tree_event(self) -> Optional[PPHostTreeEvent]:
        if not self.pp_pending_host_tree_events:
            return None
        return self.pp_pending_host_tree_events.popleft()

    def _drain_single_revoke_req(
        self,
        req_id: str,
        zero_hit: bool = False,
        mark_local_revoke: bool = True,
    ) -> None:
        self.clear_follow_rank_prefetch_issue_pending(req_id)
        if req_id in self.pp_authoritative_revoked_req_ids:
            mark_local_revoke = False
        if req_id in self.pp_soft_skipped_req_ids:
            mark_local_revoke = False
            self.pp_soft_skipped_req_ids.discard(req_id)
        had_ongoing = req_id in self.ongoing_prefetch
        loaded_tokens_before = self.prefetch_loaded_tokens_by_reqid.get(req_id, 0)
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is not None:
            last_host_node, token_ids, _, _ = info
            last_host_node.release_host()
            self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
            if self.cache_controller.prefetch_tokens_occupied < 0:
                self.cache_controller.prefetch_tokens_occupied = 0
        self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
        if zero_hit:
            self.zero_hit_prefetch_req_ids.add(req_id)
            if mark_local_revoke and self._pp_downstream_sync_enabled():
                if req_id not in self.pp_locally_revoked_req_ids:
                    self.pp_locally_revoked_req_ids.add(req_id)
                    self.pp_locally_revoked_req_queue.append(req_id)
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePrefetchCleanup] rid=%s zero_hit=%s mark_local_revoke=%s had_ongoing=%s ongoing_after=%s loaded_tokens_before=%s loaded_tokens_after=%s zero_hit_marked=%s",
                req_id,
                zero_hit,
                mark_local_revoke,
                had_ongoing,
                req_id in self.ongoing_prefetch,
                loaded_tokens_before,
                self.prefetch_loaded_tokens_by_reqid.get(req_id, 0),
                req_id in self.zero_hit_prefetch_req_ids,
            )
            logger.warning(
                "[PPReqPhase] pp=%s cp=%s tp=%s rid=%s phase=prefetch_cleanup zero_hit=%s",
                self.pp_rank,
                self.attn_cp_rank,
                _safe_attn_tp_rank(self),
                req_id,
                zero_hit,
            )

    def peek_pp_locally_revoked_req(self) -> Optional[str]:
        while self.pp_locally_revoked_req_queue:
            rid = self.pp_locally_revoked_req_queue[0]
            if rid in self.pp_retry_prefetch_req_ids:
                self.pp_locally_revoked_req_ids.discard(rid)
                self.pp_locally_revoked_req_queue.popleft()
                continue
            if rid in self.pp_locally_revoked_req_ids:
                return rid
            self.pp_locally_revoked_req_queue.popleft()
        return None

    def discard_pp_locally_revoked_req(self, req_id: str) -> None:
        self.pp_locally_revoked_req_ids.discard(req_id)
        self.pp_soft_skipped_req_ids.discard(req_id)
        while self.pp_locally_revoked_req_queue:
            rid = self.pp_locally_revoked_req_queue[0]
            if rid in self.pp_locally_revoked_req_ids:
                break
            self.pp_locally_revoked_req_queue.popleft()

    def _release_request_ephemeral_state(self, req_id: str) -> None:
        """Clear per-request PP/HiCache bookkeeping after the request is done.

        This must only run on terminal request paths (finished output or abort).
        Some suppress markers intentionally survive intermediate scheduler steps
        to avoid revoke re-hardening races, so they should not be cleared earlier.
        """
        self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
        self.prefetch_issue_count_by_reqid.pop(req_id, None)
        self.zero_hit_prefetch_req_ids.discard(req_id)
        self.pp_retry_prefetch_req_ids.discard(req_id)
        self.pp_authoritative_revoked_req_ids.discard(req_id)
        self.pp_soft_skipped_req_ids.discard(req_id)
        self.pp_staged_prefetch_skip_req_ids.discard(req_id)
        self.clear_follow_rank_prefetch_issue_pending(req_id)
        self.discard_pp_locally_revoked_req(req_id)
        self._purge_matching_local_revoke_residue(req_id)

    def release_finished_request(self, rid: str) -> None:
        self._release_request_ephemeral_state(rid)

    def _purge_matching_local_revoke_residue(self, req_id: str) -> tuple[int, int]:
        purged_deferred = 0
        if self.pp_deferred_revoke_req_ids:
            kept_items = deque()
            for deferred_req_id, deferred_zero_hit in self.pp_deferred_revoke_req_ids:
                if deferred_req_id == req_id:
                    purged_deferred += 1
                    continue
                kept_items.append((deferred_req_id, deferred_zero_hit))
            self.pp_deferred_revoke_req_ids = kept_items

        purged_queue = 0
        revoke_queue = self.cache_controller.prefetch_revoke_queue
        if hasattr(revoke_queue, "mutex") and hasattr(revoke_queue, "queue"):
            with revoke_queue.mutex:
                kept_items = deque()
                while revoke_queue.queue:
                    item = revoke_queue.queue.popleft()
                    queued_req_id, queued_zero_hit = (
                        item if isinstance(item, tuple) else (item, False)
                    )
                    if queued_req_id == req_id:
                        purged_queue += 1
                        continue
                    kept_items.append(
                        (queued_req_id, queued_zero_hit)
                        if isinstance(item, tuple)
                        else queued_req_id
                    )
                revoke_queue.queue.extend(kept_items)
                if purged_queue > 0 and hasattr(revoke_queue, "unfinished_tasks"):
                    revoke_queue.unfinished_tasks = max(
                        0, revoke_queue.unfinished_tasks - purged_queue
                    )
                if purged_queue > 0 and hasattr(revoke_queue, "not_full"):
                    revoke_queue.not_full.notify_all()

        if (purged_deferred or purged_queue) and self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPReplay][revoke_purge_local_residue] rid=%s purged_deferred=%s purged_queue=%s",
                req_id,
                purged_deferred,
                purged_queue,
            )

        return purged_deferred, purged_queue

    def _try_replay_prefetch_skip_event(self, event: PPHostTreeEvent) -> bool:
        req_id = event.rid
        if req_id is None:
            return False
        self.clear_follow_rank_prefetch_issue_pending(req_id)
        self.pp_soft_skipped_req_ids.add(req_id)
        if req_id in self.ongoing_prefetch:
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPReplay][prefetch_skip_apply] rid=%s action=cleanup_ongoing_soft",
                    req_id,
                )
            self._drain_single_revoke_req(
                req_id,
                zero_hit=True,
                mark_local_revoke=False,
            )
            return True
        self.zero_hit_prefetch_req_ids.add(req_id)
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPReplay][prefetch_skip_apply] rid=%s action=mark_zero_hit_soft",
                req_id,
            )
        return True

    def _try_replay_revoke_event(self, event: PPHostTreeEvent) -> bool:
        req_id = event.rid
        if req_id is None:
            return False

        # Treat upstream REVOKE as authoritative for downstream PP ranks.
        # Waiting for the local revoke queue to independently produce the same
        # req can leave the request stuck in wait_complete while the upstream
        # rank has already bypassed/revoked the prefetch.
        if req_id in self.ongoing_prefetch:
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPReplay][revoke_apply] rid=%s source=upstream ongoing_before=%s loaded_tokens=%s",
                    req_id,
                    True,
                    self.prefetch_loaded_tokens_by_reqid.get(req_id, 0),
                )
            self.pp_authoritative_revoked_req_ids.add(req_id)
            self._purge_matching_local_revoke_residue(req_id)
            self._drain_single_revoke_req(
                req_id,
                zero_hit=True,
                mark_local_revoke=False,
            )
            self.discard_pp_locally_revoked_req(req_id)
            return True

        if self.pp_deferred_revoke_req_ids:
            deferred_req_ids = [rid for rid, _ in self.pp_deferred_revoke_req_ids]
            if req_id in deferred_req_ids:
                deferred_items = list(self.pp_deferred_revoke_req_ids)
                self.pp_deferred_revoke_req_ids.clear()
                matched_zero_hit = False
                skipped = 0
                for deferred_req_id, deferred_zero_hit in deferred_items:
                    if deferred_req_id == req_id:
                        matched_zero_hit = deferred_zero_hit
                        continue
                    skipped += 1
                    self.pp_deferred_revoke_req_ids.append(
                        (deferred_req_id, deferred_zero_hit)
                    )
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPReplay][revoke_apply] rid=%s source=deferred_queue zero_hit=%s skipped_unrelated=%s",
                        req_id,
                        matched_zero_hit,
                        skipped,
                    )
                self._drain_single_revoke_req(req_id, zero_hit=matched_zero_hit)
                self.discard_pp_locally_revoked_req(req_id)
                return True

            deferred_head = deferred_req_ids[0]
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPReplay][revoke_wait] rid=%s reason=deferred_head_mismatch deferred_head=%s deferred_pending=%s",
                    req_id,
                    deferred_head,
                    deferred_req_ids[:4],
                )

        scanned_unrelated = []
        while True:
            try:
                queued_item = self.cache_controller.prefetch_revoke_queue.get_nowait()
            except Empty:
                for queued_req_id, queued_zero_hit in scanned_unrelated:
                    self.pp_deferred_revoke_req_ids.append((queued_req_id, queued_zero_hit))
                # Upstream REVOKE is authoritative. If the downstream rank has
                # already quiesced this request locally (no ongoing prefetch and
                # no matching local revoke item to drain), treat the revoke as an
                # idempotent no-op so it does not become a permanent PP queue
                # head blocker.
                if req_id not in self.ongoing_prefetch:
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePPReplay][revoke_apply] rid=%s source=noop_quiescent scanned_unrelated=%s",
                            req_id,
                            [rid for rid, _ in scanned_unrelated[:4]],
                        )
                    self._purge_matching_local_revoke_residue(req_id)
                    self.discard_pp_locally_revoked_req(req_id)
                    self.zero_hit_prefetch_req_ids.add(req_id)
                    self.pp_authoritative_revoked_req_ids.add(req_id)
                    return True
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPReplay][revoke_wait] rid=%s reason=no_local_revoke_queue ongoing=%s scanned_unrelated=%s",
                        req_id,
                        req_id in self.ongoing_prefetch,
                        [rid for rid, _ in scanned_unrelated[:4]],
                    )
                return False
            queued_req_id, queued_zero_hit = (
                queued_item if isinstance(queued_item, tuple) else (queued_item, False)
            )

            if queued_req_id != req_id:
                scanned_unrelated.append((queued_req_id, queued_zero_hit))
                continue

            for unrelated_req_id, unrelated_zero_hit in scanned_unrelated:
                self.pp_deferred_revoke_req_ids.append(
                    (unrelated_req_id, unrelated_zero_hit)
                )
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPReplay][revoke_apply] rid=%s source=local_revoke_queue zero_hit=%s skipped_unrelated=%s",
                    queued_req_id,
                    queued_zero_hit,
                    len(scanned_unrelated),
                )
            self._drain_single_revoke_req(queued_req_id, zero_hit=queued_zero_hit)
            self.discard_pp_locally_revoked_req(queued_req_id)
            return True

    def _write_commit_event_matches(
        self,
        event: PPHostTreeEvent,
        nodes: List[TreeNode],
    ) -> bool:
        if len(nodes) != len(event.node_key_lens):
            return False
        for idx, node in enumerate(nodes):
            if len(node.key) != event.node_key_lens[idx]:
                return False
            if node.key.extra_key != event.node_extra_keys[idx]:
                return False
            if node.get_last_hash_value() != event.node_last_hashes[idx]:
                return False
        return True

    def _consume_write_ack_group(
        self, finish_event, ack_list: List[int], emit_event: bool
    ) -> None:
        finish_event.synchronize()
        committed_nodes: List[TreeNode] = []
        for ack_id in ack_list:
            backuped_node = self.ongoing_write_through.pop(ack_id)
            committed_nodes.append(backuped_node)
            self.dec_lock_ref(backuped_node)
            if self.enable_storage:
                self.write_backup_storage(backuped_node)
        self._write_backup_commit_nodes += len(committed_nodes)
        if emit_event:
            event = PPHostTreeEvent(
                seq=self._next_pp_host_tree_seq(),
                kind="WRITE_BACKUP_COMMITTED",
                node_ids=[node.id for node in committed_nodes],
                node_key_lens=[len(node.key) for node in committed_nodes],
                node_last_hashes=[
                    node.get_last_hash_value() for node in committed_nodes
                ],
                node_extra_keys=[node.key.extra_key for node in committed_nodes],
                node_parent_key_lens=[
                    len(node.parent.key)
                    if node.parent is not None and node.parent.key is not None
                    else 0
                    for node in committed_nodes
                ],
                node_parent_last_hashes=[
                    node.parent.get_last_hash_value()
                    if node.parent is not None
                    else None
                    for node in committed_nodes
                ],
                node_parent_extra_keys=[
                    node.parent.key.extra_key
                    if node.parent is not None and node.parent.key is not None
                    else None
                    for node in committed_nodes
                ],
            )
            committed_count = len(committed_nodes)
            estimated_bytes = _estimate_pp_write_backup_event_bytes(event)
            self._write_backup_event_count += 1
            self._write_backup_event_total_nodes += committed_count
            self._write_backup_event_max_nodes = max(
                self._write_backup_event_max_nodes, committed_count
            )
            self._write_backup_event_estimated_bytes += estimated_bytes
            self._write_backup_event_max_estimated_bytes = max(
                self._write_backup_event_max_estimated_bytes, estimated_bytes
            )
            self._append_pp_host_tree_event(event)

    def _try_replay_write_backup_event(self, event: PPHostTreeEvent) -> bool:
        miss_recorded = False

        def _record_miss() -> None:
            nonlocal miss_recorded
            if not miss_recorded:
                self._write_backup_replay_miss += 1
                miss_recorded = True

        if self.cache_controller.ack_write_queue:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue[0]
            if finish_event.query():
                nodes = [self.ongoing_write_through.get(ack_id) for ack_id in ack_list]
                if (
                    not any(node is None for node in nodes)
                    and self._write_commit_event_matches(event, nodes)
                ):
                    self.cache_controller.ack_write_queue.pop(0)
                    self._consume_write_ack_group(
                        finish_event, ack_list, emit_event=True
                    )
                    self._write_backup_replay_apply_local_ack += 1
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePPEvent][replay_apply_write_backup] pp=%s cp=%s seq=%s source=local_ack rid=%s nodes=%s",
                            self.pp_rank,
                            self.attn_cp_rank,
                            event.seq,
                            event.rid,
                            len(ack_list),
                        )
                    return True
            else:
                _record_miss()
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPEvent][replay_write_backup_miss] pp=%s cp=%s seq=%s reason=ack_not_ready ack_head=%s ongoing_write=%s event_nodes=%s",
                        self.pp_rank,
                        self.attn_cp_rank,
                        event.seq,
                        ack_list[:8],
                        len(self.ongoing_write_through),
                        len(event.node_key_lens),
                    )
        else:
            _record_miss()
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPEvent][replay_write_backup_miss] pp=%s cp=%s seq=%s reason=no_ack_queue ongoing_write=%s event_nodes=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    event.seq,
                    len(self.ongoing_write_through),
                    len(event.node_key_lens),
                )

        if not event.node_key_lens:
            _record_miss()
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPEvent][replay_write_backup_miss] pp=%s cp=%s seq=%s reason=empty_event_nodes ongoing_write=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    event.seq,
                    len(self.ongoing_write_through),
                )
            return False

        ack_list = None
        nodes = None
        removed_ack_idx = None
        for idx, (_, _, queued_ack_list) in enumerate(self.cache_controller.ack_write_queue):
            queued_nodes = [
                self.ongoing_write_through.get(ack_id) for ack_id in queued_ack_list
            ]
            if any(node is None for node in queued_nodes):
                continue
            if not self._write_commit_event_matches(event, queued_nodes):
                continue
            removed_ack_idx = idx
            ack_list = queued_ack_list
            nodes = queued_nodes
            break

        if ack_list is None or nodes is None:
            if len(event.node_ids) == len(event.node_key_lens):
                queued_nodes = [
                    self.ongoing_write_through.get(node_id) for node_id in event.node_ids
                ]
                if (
                    not any(node is None for node in queued_nodes)
                    and self._write_commit_event_matches(event, queued_nodes)
                ):
                    ack_list = list(event.node_ids)
                    nodes = queued_nodes
        if ack_list is None or nodes is None:
            sample_nodes = list(self.ongoing_write_through.items())[:4]
            _record_miss()
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPEvent][replay_write_backup_miss] pp=%s cp=%s seq=%s reason=no_matching_nodes event_key_lens=%s event_hashes=%s sample_ongoing=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    event.seq,
                    event.node_key_lens[:8],
                    event.node_last_hashes[:4],
                    [
                        (
                            node_id,
                            len(node.key),
                            node.get_last_hash_value(),
                            node.key.extra_key,
                        )
                        for node_id, node in sample_nodes
                    ],
                )
            return False

        if removed_ack_idx is not None:
            self.cache_controller.ack_write_queue.pop(removed_ack_idx)

        for node_id in ack_list:
            backuped_node = self.ongoing_write_through.pop(node_id)
            self.dec_lock_ref(backuped_node)

        self._write_backup_replay_apply_authoritative += 1
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePPEvent][replay_apply_write_backup] pp=%s cp=%s seq=%s source=authoritative_event rid=%s nodes=%s removed_ack=%s",
                self.pp_rank,
                self.attn_cp_rank,
                event.seq,
                event.rid,
                len(event.node_ids),
                removed_ack_idx is not None,
            )
        return True

    def _finalize_prefetch_progress(
        self, req_id: str, operation: PrefetchOperation, emit_event: bool
    ) -> int:
        last_host_node, token_ids, host_indices, _ = self.ongoing_prefetch[req_id]
        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()

        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            req_id=req_id,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
        self.zero_hit_prefetch_req_ids.discard(req_id)

        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
            logger.warning(
                "[HiCacheMatchChain] prefetch finalize: rid=%s completed_tokens=%s "
                "min_completed_tokens=%s matched_length=%s loaded_from_storage=%s "
                "pp=%s cp=%s",
                req_id,
                completed_tokens,
                min_completed_tokens,
                matched_length,
                loaded_from_storage,
                self.pp_rank,
                self.attn_cp_rank,
            )
        if os.getenv("SGLANG_DEBUG_HICACHE_HOST_DRIFT", "0") == "1":
            anchor_hash = (
                last_host_node.get_last_hash_value() if last_host_node is not None else None
            )
            first_suffix_token = (
                fetched_token_ids[matched_length]
                if matched_length < len(fetched_token_ids)
                else None
            )
            first_suffix_hash = (
                hash_value[matched_length // self.page_size]
                if matched_length // self.page_size < len(hash_value)
                else None
            )
            logger.warning(
                "[HiCacheFinalizeInsert] rid=%s pp=%s cp=%s anchor_node=%s anchor_hash=%s "
                "token_len=%s completed=%s matched=%s loaded=%s first_suffix_token=%s "
                "first_suffix_hash=%s",
                req_id,
                self.pp_rank,
                self.attn_cp_rank,
                last_host_node.id if last_host_node is not None else None,
                anchor_hash,
                len(token_ids),
                min_completed_tokens,
                matched_length,
                loaded_from_storage,
                first_suffix_token,
                first_suffix_hash,
            )
            logger.warning(
                "[PPReqPhase] pp=%s cp=%s tp=%s rid=%s phase=prefetch_finalize completed=%s matched=%s loaded=%s anchor_node=%s",
                self.pp_rank,
                self.attn_cp_rank,
                _safe_attn_tp_rank(self),
                req_id,
                min_completed_tokens,
                matched_length,
                loaded_from_storage,
                last_host_node.id if last_host_node is not None else None,
            )

        if emit_event:
            self._append_pp_host_tree_event(
                PPHostTreeEvent(
                    seq=self._next_pp_host_tree_seq(),
                    kind="PREFETCH_FINALIZE",
                    rid=req_id,
                    loaded_from_storage=loaded_from_storage,
                )
            )
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return loaded_from_storage

    def _try_replay_prefetch_finalize_event(self, event: PPHostTreeEvent) -> bool:
        req_id = event.rid
        if req_id is None:
            return False
        if req_id not in self.ongoing_prefetch:
            if event.loaded_from_storage > 0:
                self.zero_hit_prefetch_req_ids.discard(req_id)
                self.discard_pp_locally_revoked_req(req_id)
                self.pp_retry_prefetch_req_ids.add(req_id)
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPEvent][replay_mark_retry_prefetch] pp=%s cp=%s seq=%s rid=%s loaded=%s",
                        self.pp_rank,
                        self.attn_cp_rank,
                        event.seq,
                        req_id,
                        event.loaded_from_storage,
                    )
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePPEvent][replay_drop_stale_finalize] pp=%s cp=%s seq=%s rid=%s loaded=%s zero_hit=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    event.seq,
                    req_id,
                    event.loaded_from_storage,
                    req_id in self.zero_hit_prefetch_req_ids,
                )
            return True
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return False
        if not self.can_terminate_prefetch(operation):
            # Upstream may already have finalized an empty prefetch (no hash pages /
            # no completed tokens). In that case, waiting for local
            # can_terminate_prefetch() will never make progress, so consume the
            # authoritative finalize by cleaning up the local empty prefetch state.
            if (
                len(operation.hash_value) == 0
                and operation.completed_tokens == 0
                and event.loaded_from_storage == 0
            ):
                last_host_node.release_host()
                del self.ongoing_prefetch[req_id]
                self.cache_controller.append_host_mem_release(host_indices)
                self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
                if self.cache_controller.prefetch_tokens_occupied < 0:
                    self.cache_controller.prefetch_tokens_occupied = 0
                self.prefetch_loaded_tokens_by_reqid[req_id] = 0
                self.zero_hit_prefetch_req_ids.discard(req_id)
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPReplay][finalize_empty_apply] rid=%s upstream_loaded=%s",
                        req_id,
                        event.loaded_from_storage,
                    )
                return True
            return False
        loaded_from_storage = self._finalize_prefetch_progress(
            req_id, operation, emit_event=True
        )
        if loaded_from_storage != event.loaded_from_storage:
            logger.warning(
                "[PPHiCacheSync] prefetch finalize mismatch: rid=%s upstream=%s local=%s "
                "pp=%s cp=%s",
                req_id,
                event.loaded_from_storage,
                loaded_from_storage,
                self.pp_rank,
                self.attn_cp_rank,
            )
        return True

    def replay_pp_host_tree_events(self) -> int:
        if not self._pp_downstream_sync_enabled():
            return 0
        if self._in_pp_host_tree_replay:
            return 0

        replayed = 0
        self._in_pp_host_tree_replay = True
        try:
            while self.pp_pending_host_tree_events:
                event = self.pp_pending_host_tree_events[0]
                if (
                    event.kind == "WRITE_BACKUP_COMMITTED"
                    and not self._pp_write_backup_replay_enabled()
                ):
                    self.pp_pending_host_tree_events.popleft()
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePPEvent][replay_skip] pp=%s cp=%s seq=%s kind=%s rid=%s reason=write_backup_replay_disabled pending_after=%s",
                            self.pp_rank,
                            self.attn_cp_rank,
                            event.seq,
                            event.kind,
                            event.rid,
                            len(self.pp_pending_host_tree_events),
                        )
                    replayed += 1
                    continue
                progressed = False
                if event.kind == "WRITE_BACKUP_COMMITTED":
                    progressed = self._try_replay_write_backup_event(event)
                elif event.kind == "PREFETCH_SKIP":
                    progressed = self._try_replay_prefetch_skip_event(event)
                elif event.kind == "REVOKE":
                    progressed = self._try_replay_revoke_event(event)
                elif event.kind == "PREFETCH_FINALIZE":
                    progressed = self._try_replay_prefetch_finalize_event(event)
                if not progressed:
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePPEvent][replay_blocked] pp=%s cp=%s seq=%s kind=%s rid=%s pending=%s ongoing=%s zero_hit=%s",
                            self.pp_rank,
                            self.attn_cp_rank,
                            event.seq,
                            event.kind,
                            event.rid,
                            len(self.pp_pending_host_tree_events),
                            event.rid in self.ongoing_prefetch if event.rid is not None else False,
                            event.rid in self.zero_hit_prefetch_req_ids if event.rid is not None else False,
                        )
                    break
                self.pp_pending_host_tree_events.popleft()
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePPEvent][replay_applied] pp=%s cp=%s seq=%s kind=%s rid=%s pending_after=%s ongoing=%s zero_hit=%s loaded=%s",
                        self.pp_rank,
                        self.attn_cp_rank,
                        event.seq,
                        event.kind,
                        event.rid,
                        len(self.pp_pending_host_tree_events),
                        event.rid in self.ongoing_prefetch if event.rid is not None else False,
                        event.rid in self.zero_hit_prefetch_req_ids if event.rid is not None else False,
                        self.prefetch_loaded_tokens_by_reqid.get(event.rid, 0)
                        if event.rid is not None
                        else 0,
                    )
                replayed += 1
        finally:
            self._in_pp_host_tree_replay = False
        return replayed

    def _try_fast_forward_revoke_for_req(self, req_id: str) -> bool:
        """Apply an already-arrived REVOKE for req_id when it is only blocked by
        unrelated PREFETCH_FINALIZE events ahead of it.

        This keeps authoritative upstream revoke semantics while avoiding
        cross-request HOL blocking from finalize events that do not affect the
        current request.
        """
        if not self._pp_downstream_sync_enabled() or not self.pp_pending_host_tree_events:
            return False

        skipped_unrelated_finalize = 0
        skipped_unrelated_revoke = 0
        for event in self.pp_pending_host_tree_events:
            if event.kind == "PREFETCH_FINALIZE" and event.rid != req_id:
                skipped_unrelated_finalize += 1
                continue
            if event.kind == "REVOKE" and event.rid != req_id:
                skipped_unrelated_revoke += 1
                continue
            if event.kind == "REVOKE" and event.rid == req_id:
                if self._try_replay_revoke_event(event):
                    self.pp_pending_host_tree_events.remove(event)
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePPReplay][revoke_fast_apply] rid=%s skipped_unrelated_finalize=%s skipped_unrelated_revoke=%s",
                            req_id,
                            skipped_unrelated_finalize,
                            skipped_unrelated_revoke,
                        )
                    return True
                return False
            return False

        return False

    def _pp_event_blocks_prefetch_progress(
        self, event: PPHostTreeEvent, req_id: str
    ) -> bool:
        # Request-scoped PP events should only block the matching request. Global
        # structural events are handled by dedicated scheduler barriers instead of
        # stalling unrelated requests here.
        if event.kind == "WRITE_BACKUP_COMMITTED":
            return False
        if event.kind in ("PREFETCH_SKIP", "REVOKE", "PREFETCH_FINALIZE"):
            return event.rid == req_id
        return True

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, NSATokenToKVPool):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCacheWriteBackup] pp=%s cp=%s node_id=%s key_len=%s last_hash=%s extra_key=%s write_back=%s ongoing_write=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    node.id,
                    len(node.key),
                    node.get_last_hash_value(),
                    node.key.extra_key,
                    write_back,
                    len(self.ongoing_write_through),
                )
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCacheWriteBackup] pp=%s cp=%s node_id=%s action=alloc_failed key_len=%s last_hash=%s extra_key=%s",
                    self.pp_rank,
                    self.attn_cp_rank,
                    node.id,
                    len(node.key),
                    node.get_last_hash_value(),
                    node.key.extra_key,
                )
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        if (
            self._pp_downstream_sync_enabled()
            and self._pp_write_backup_replay_enabled()
        ):
            while self.pp_pending_host_tree_events:
                event = self.pp_pending_host_tree_events[0]
                if event.kind != "WRITE_BACKUP_COMMITTED":
                    break
                if not self._try_replay_write_backup_event(event):
                    break
                self.pp_pending_host_tree_events.popleft()
            return

        finish_count = 0
        for _, finish_event, _ in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce_attn_groups(queue_size, torch.distributed.ReduceOp.MIN)

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            self._consume_write_ack_group(finish_event, ack_list, emit_event=True)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def _to_radix_key(self, token_ids: List[int]) -> RadixKey:
        """Convert raw token_ids to a RadixKey for tree walking.

        Must use list (not tuple) to match scheduler's RadixKey format,
        since _key_match_paged compares slices directly and list != tuple.
        """
        return RadixKey(token_ids=list(token_ids))

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU demotion: no BlockRemoved since block is still reachable via load_back
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._tree_delete_regular_leaf += 1
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit BlockRemoved so the router removes this block from its index.
            self._record_remove_event(x)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            self._tree_delete_host_leaf += 1
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def sync_hicache_attn_groups(self) -> None:
        self._barrier_attn_groups()

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def _get_prefetch_progress_debug(self, req_id: str) -> dict[str, Any]:
        if req_id not in self.ongoing_prefetch:
            return {"ongoing": False}

        _, token_ids, host_indices, operation = self.ongoing_prefetch[req_id]
        expected_tokens = len(operation.hash_value) * self.page_size
        completed = (
            False
            if len(operation.hash_value) == 0
            else operation.completed_tokens == expected_tokens
        )
        return {
            "ongoing": True,
            "policy": self.prefetch_stop_policy,
            "host_indices_none": operation.host_indices is None,
            "completed_tokens": operation.completed_tokens,
            "expected_tokens": expected_tokens,
            "hash_pages": len(operation.hash_value),
            "token_ids": len(token_ids),
            "completed": completed,
            "terminated": operation.is_terminated(),
            "age_sec": round(time.monotonic() - operation.start_time, 3),
            "loaded_tokens": self.prefetch_loaded_tokens_by_reqid.get(req_id, 0),
        }

    def get_prefetch_progress_debug(self, req_id: str) -> dict[str, Any]:
        return self._get_prefetch_progress_debug(req_id)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        if self._pp_downstream_sync_enabled() and not self._in_pp_host_tree_replay:
            self.replay_pp_host_tree_events()
            if req_id not in self.ongoing_prefetch:
                # The ordered PP replay may have already revoked/finalized this request
                # on the local rank. Treat it as completed for the scheduler path.
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePrefetchWaitResolved] rid=%s reason=pp_replay_removed_ongoing",
                        req_id,
                    )
                return True
            if self._try_fast_forward_revoke_for_req(req_id):
                if req_id not in self.ongoing_prefetch:
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePrefetchWaitResolved] rid=%s reason=fast_forward_revoke",
                            req_id,
                        )
                    return True
            event = self._peek_pp_host_tree_event()
            if event is not None:
                if self._pp_event_blocks_prefetch_progress(event, req_id):
                    if self._hicache_verbose_enabled():
                        logger.warning(
                            "[HiCachePrefetchWaitBlocked] rid=%s reason=matching_pp_event event_kind=%s event_rid=%s",
                            req_id,
                            event.kind,
                            event.rid,
                        )
                    return False
                if self._hicache_verbose_enabled():
                    logger.warning(
                        "[HiCachePrefetchWaitPass] rid=%s reason=unrelated_pp_event event_kind=%s event_rid=%s event_seq=%s",
                        req_id,
                        event.kind,
                        event.rid,
                        event.seq,
                    )

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            debug_state = self._get_prefetch_progress_debug(req_id)
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCacheEmptyPrefetchState] rid=%s zero_hit_marked=%s replay_pending=%s state=%s",
                    req_id,
                    req_id in self.zero_hit_prefetch_req_ids,
                    self._peek_pp_host_tree_event() is not None
                    if self._pp_downstream_sync_enabled()
                    else False,
                    debug_state,
                )
                logger.warning(
                    "[HiCachePrefetchWait] rid=%s state=%s",
                    req_id,
                    debug_state,
                )
            return False
        self._finalize_prefetch_progress(req_id, operation, emit_event=True)
        if self._pp_downstream_sync_enabled():
            event = self._peek_pp_host_tree_event()
            if (
                event is not None
                and event.kind == "PREFETCH_FINALIZE"
                and event.rid == req_id
            ):
                self._pop_pp_host_tree_event()

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        # Keep the zero-hit marker until an explicit retry signal or request
        # teardown clears it. Otherwise the same waiting req can re-enter local
        # storage prefetch immediately after a zero-hit revoke.
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        start_time = time.perf_counter()
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        page_aligned_len = len(key)
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node, walk_steps, split_count, path_fanout_sum, path_max_fanout = self._match_prefix_helper(
            self.root_node, key
        )
        value_segments = len(value)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        host_climb_steps = 0
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
            host_climb_steps += 1
        backup_climb_steps = 0
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent
            backup_climb_steps += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._match_perf_calls += 1
        self._match_perf_total_ms += elapsed_ms
        self._match_perf_max_ms = max(self._match_perf_max_ms, elapsed_ms)
        self._match_perf_total_walk_steps += walk_steps
        self._match_perf_max_walk_steps = max(
            self._match_perf_max_walk_steps, walk_steps
        )
        self._match_perf_total_splits += split_count
        self._match_perf_max_splits = max(self._match_perf_max_splits, split_count)
        self._match_perf_total_host_climb += host_climb_steps
        self._match_perf_max_host_climb = max(
            self._match_perf_max_host_climb, host_climb_steps
        )
        self._match_perf_total_backup_climb += backup_climb_steps
        self._match_perf_max_backup_climb = max(
            self._match_perf_max_backup_climb, backup_climb_steps
        )
        self._match_perf_total_value_segments += value_segments
        self._match_perf_total_key_len += len(params.key)
        self._match_perf_total_aligned_len += page_aligned_len
        self._match_perf_total_device_hit += len(value)
        self._match_perf_total_host_hit += host_hit_length
        self._match_perf_total_path_fanout += path_fanout_sum
        self._match_perf_max_path_fanout = max(
            self._match_perf_max_path_fanout, path_max_fanout
        )
        if walk_steps == 0:
            self._match_perf_root_only_calls += 1
        if split_count > 0:
            self._match_perf_split_calls += 1
        if host_climb_steps > 0:
            self._match_perf_host_climb_calls += 1
        if backup_climb_steps > 0:
            self._match_perf_backup_climb_calls += 1
        if value_segments > 1:
            self._match_perf_multi_segment_calls += 1
        if len(value) > 0:
            self._match_perf_device_hit_calls += 1
        if host_hit_length > 0:
            self._match_perf_host_hit_calls += 1

        if (
            os.getenv("SGLANG_DEBUG_HICACHE_HOST_DRIFT", "0") == "1"
            and params.req is not None
        ):
            host_path = []
            walk_node = last_node
            while walk_node is not None and walk_node is not self.root_node and len(host_path) < 8:
                host_path.append(
                    (
                        walk_node.id,
                        len(walk_node.key) if walk_node.key is not None else 0,
                        walk_node.evicted,
                        walk_node.backuped,
                        len(walk_node.host_value) if walk_node.host_value is not None else 0,
                    )
                )
                if not walk_node.evicted:
                    break
                walk_node = walk_node.parent

            backup_walk = []
            walk_node = last_host_node
            while walk_node is not None and walk_node is not self.root_node and len(backup_walk) < 8:
                backup_walk.append(
                    (
                        walk_node.id,
                        len(walk_node.key) if walk_node.key is not None else 0,
                        walk_node.evicted,
                        walk_node.backuped,
                        len(walk_node.host_value) if walk_node.host_value is not None else 0,
                    )
                )
                if walk_node.backuped:
                    break
                walk_node = walk_node.parent

            if len(params.key) >= 400 or host_hit_length == 0:
                logger.warning(
                    "[HiCacheMatchPath] rid=%s key_len=%s device_hit=%s host_hit=%s total_cached=%s "
                    "last_device=%s last_host=%s host_path=%s backup_walk=%s pp=%s cp=%s tp=%s",
                    params.req.rid,
                    len(params.key),
                    len(value),
                    host_hit_length,
                    len(value) + host_hit_length,
                    last_node.id if last_node is not None else None,
                    last_host_node.id if last_host_node is not None else None,
                    host_path,
                    backup_walk,
                    self.pp_rank,
                    self.attn_cp_rank,
                    self.cache_controller.tp_rank,
                )

        if (
            os.getenv("SGLANG_DEBUG_HICACHE_MATCH", "0") == "1"
            and params.req is not None
        ):
            logger.warning(
                "[HiCacheMatch] rid=%s key_len=%s aligned_len=%s device_hit=%s "
                "host_hit=%s total_cached=%s page_size=%s pp=%s cp=%s tp=%s "
                "last_device=%s last_host=%s",
                params.req.rid,
                len(params.key),
                page_aligned_len,
                len(value),
                host_hit_length,
                len(value) + host_hit_length,
                self.page_size,
                self.pp_rank,
                self.attn_cp_rank,
                self.cache_controller.tp_rank,
                last_node.id if last_node is not None else None,
                last_host_node.id if last_host_node is not None else None,
            )

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.pp_authoritative_revoked_req_ids.discard(req_id)
        if req_id in self.zero_hit_prefetch_req_ids:
            # This request already proved to have no storage benefit on this pass.
            # Skip re-entering the expensive prefetch -> revoke lifecycle and
            # let it go straight through normal recompute.
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=zero_hit_marked tokens=%s threshold=%s",
                    req_id,
                    len(new_input_tokens),
                    self.prefetch_threshold,
                )
            return

        new_input_tokens = (
            convert_to_bigram_key(new_input_tokens)
            if self.is_eagle
            else new_input_tokens
        )
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if not self.enable_storage:
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=storage_disabled tokens=%s aligned_tokens=%s threshold=%s",
                    req_id,
                    len(new_input_tokens),
                    prefetch_length,
                    self.prefetch_threshold,
                )
            return
        if prefetch_length < self.prefetch_threshold:
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=below_threshold tokens=%s aligned_tokens=%s threshold=%s",
                    req_id,
                    len(new_input_tokens),
                    prefetch_length,
                    self.prefetch_threshold,
                )
            if self._pp_downstream_sync_enabled():
                self._append_pp_host_tree_event(
                    PPHostTreeEvent(
                        seq=self._next_pp_host_tree_seq(),
                        kind="PREFETCH_SKIP",
                        rid=req_id,
                    )
                )
            return
        if self._pp_should_skip_large_shallow_prefetch(
            last_host_node, prefetch_length
        ):
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=pp_first_rank_defer_large_shallow "
                    "tokens=%s aligned_tokens=%s anchor_node=%s anchor_key_len=%s threshold=%s",
                    req_id,
                    len(new_input_tokens),
                    prefetch_length,
                    last_host_node.id if last_host_node is not None else None,
                    len(last_host_node.key) if last_host_node is not None and last_host_node.key is not None else 0,
                    self.prefetch_threshold,
                )
            self._append_pp_host_tree_event(
                PPHostTreeEvent(
                    seq=self._next_pp_host_tree_seq(),
                    kind="PREFETCH_SKIP",
                    rid=req_id,
                )
            )
            return
        if self.cache_controller.prefetch_rate_limited():
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=rate_limited tokens=%s aligned_tokens=%s threshold=%s occupied=%s",
                    req_id,
                    len(new_input_tokens),
                    prefetch_length,
                    self.prefetch_threshold,
                    self.cache_controller.prefetch_tokens_occupied,
                )
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            last_host_node.release_host()
            # no sufficient host memory for prefetch
            if self._hicache_verbose_enabled():
                logger.warning(
                    "[HiCachePrefetchDecision] rid=%s action=skip reason=host_alloc_failed aligned_tokens=%s threshold=%s occupied=%s",
                    req_id,
                    prefetch_length,
                    self.prefetch_threshold,
                    self.cache_controller.prefetch_tokens_occupied,
                )
            return
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        issue_idx = self.prefetch_issue_count_by_reqid.get(req_id, 0) + 1
        self.prefetch_issue_count_by_reqid[req_id] = issue_idx
        if os.getenv("SGLANG_DEBUG_PP_PREFETCH_TRACE", "0") == "1":
            logger.warning(
                "[PPPrefetchTrace] pp=%s cp=%s tp=%s rid=%s phase=issue issue_idx=%s "
                "aligned_tokens=%s anchor_node=%s anchor_backuped=%s last_hash=%s prefix_keys=%s",
                self.pp_rank,
                self.attn_cp_rank,
                _safe_attn_tp_rank(self),
                req_id,
                issue_idx,
                prefetch_length,
                last_host_node.id if last_host_node is not None else None,
                last_host_node.backuped if last_host_node is not None else None,
                last_hash,
                0 if prefix_keys is None else len(prefix_keys),
            )
        if self._hicache_verbose_enabled():
            logger.warning(
                "[PPReqPhase] pp=%s cp=%s tp=%s rid=%s phase=prefetch_issue token_count=%s last_hash=%s prefix_keys=%s",
                self.pp_rank,
                self.attn_cp_rank,
                _safe_attn_tp_rank(self),
                req_id,
                len(new_input_tokens),
                last_hash,
                0 if prefix_keys is None else len(prefix_keys),
            )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)
        if self._hicache_verbose_enabled():
            logger.warning(
                "[HiCachePrefetchDecision] rid=%s action=issue aligned_tokens=%s threshold=%s occupied=%s",
                req_id,
                len(new_input_tokens),
                self.prefetch_threshold,
                self.cache_controller.prefetch_tokens_occupied,
            )

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value, req_id: str | None = None
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                self._tree_alloc_host_insert_split += 1
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            self._tree_alloc_host_leaf += 1
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if os.getenv("SGLANG_DEBUG_HICACHE_HOST_DRIFT", "0") == "1":
                logger.warning(
                    "[HiCacheHostInsert] rid=%s pp=%s cp=%s parent=%s node=%s key_len=%s "
                    "host_len=%s hash_pages=%s extra_key=%s child_key=%s first_token=%s "
                    "first_hash=%s",
                    req_id,
                    self.pp_rank,
                    self.attn_cp_rank,
                    node.id if node is not None else None,
                    new_node.id,
                    len(new_node.key) if new_node.key is not None else 0,
                    len(new_node.host_value) if new_node.host_value is not None else 0,
                    len(new_node.hash_value) if new_node.hash_value is not None else 0,
                    new_node.key.extra_key if new_node.key is not None else None,
                    child_key,
                    new_node.key.token_ids[0]
                    if new_node.key is not None and len(new_node.key.token_ids) > 0
                    else None,
                    new_node.hash_value[0]
                    if new_node.hash_value is not None and len(new_node.hash_value) > 0
                    else None,
                )

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []
        walk_steps = 0
        split_count = 0
        path_fanout_sum = 0
        path_max_fanout = 0

        while len(key) > 0 and child_key in node.children.keys():
            current_fanout = len(node.children)
            path_fanout_sum += current_fanout
            path_max_fanout = max(path_max_fanout, current_fanout)
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            walk_steps += 1
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self._tree_alloc_match_split += 1
                split_count += 1
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node, walk_steps, split_count, path_fanout_sum, path_max_fanout

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        if os.getenv("SGLANG_DEBUG_HICACHE_HOST_DRIFT", "0") == "1":
            logger.warning(
                "[HiCacheNodeSplit] pp=%s cp=%s parent=%s new_node=%s child=%s split_len=%s "
                "new_key_len=%s child_key_len=%s new_has_host=%s child_has_host=%s "
                "new_hash_pages=%s child_hash_pages=%s",
                self.pp_rank,
                self.attn_cp_rank,
                new_node.parent.id if new_node.parent is not None else None,
                new_node.id,
                child.id,
                split_len,
                len(new_node.key) if new_node.key is not None else 0,
                len(child.key) if child.key is not None else 0,
                new_node.host_value is not None,
                child.host_value is not None,
                len(new_node.hash_value) if new_node.hash_value is not None else 0,
                len(child.hash_value) if child.hash_value is not None else 0,
            )

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                self._tree_alloc_insert_split += 1
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            self._tree_alloc_device_leaf += 1
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up per-request transient PP/HiCache state for aborted requests.
        self._release_request_ephemeral_state(rid)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
