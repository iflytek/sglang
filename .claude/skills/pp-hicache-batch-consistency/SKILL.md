---
name: pp-hicache-batch-consistency
description: Guide to diagnosing and fixing PP (Pipeline Parallel) batch pick inconsistency in SGLang's HiRadixCache (L3 storage) system. Use when debugging NCCL hangs caused by PP0/PP1 batch composition mismatches, tree state divergence, or prefix match inconsistencies in multi-stage pipeline parallel with HiCache enabled.
---

# PP HiCache Batch Pick Consistency

## Problem Statement

In SGLang's Pipeline Parallel (PP) mode with HiRadixCache (L3 storage), PP0 and PP1 independently pick batches using `get_new_batch_prefill()`. Both ranks maintain their own radix tree (device tree + host tree), but asynchronous L3 operations (write-back, prefetch, eviction) cause their tree states to diverge. When batch compositions don't match, NCCL collective operations hang because the two ranks expect different tensor shapes.

**Symptom**: NCCL timeout/hang during PP forward pass. PP0 picks a batch with different requests or different prefix lengths than PP1.

## Architecture: PP Event Loop Timing

Understanding the event loop order is critical. The two PP ranks have asymmetric processing order:

```
PP0 (first rank):
  recv_requests() → process_input_requests()
  → get_next_batch_to_run()           ← batch pick happens HERE
    → check_prefetch_progress()        ← may finalize prefetch, emit events
    → init_next_round_input()          ← match_prefix, tree walk
    → writing_check()                  ← may consume write acks
  → forward pass (GPU)
  → _pp_build_req_payload()            ← consume outgoing events, send to PP1
    → consume_pp_host_tree_events()    ← drains pp_outgoing_host_tree_events

PP1 (last rank):
  recv from PP0                        ← receives events
  → _pp_apply_hicache_sync_before_batch()  ← enqueue events
  → get_next_batch_to_run()            ← batch pick happens HERE
    → replay_pp_host_tree_events()     ← apply queued events
    → check_prefetch_progress()
    → init_next_round_input()
    → writing_check()
```

**Key insight**: PP0's batch pick happens BEFORE `pp_outgoing_host_tree_events` is consumed/sent. PP1's batch pick happens AFTER events are enqueued. This natural timing means no explicit ack mechanism is needed — barriers naturally chain across iterations.

## Event Types

| Event | Has RID | Effect on Tree |
|-------|---------|---------------|
| `WRITE_BACKUP_COMMITTED` | No (node-level) | Marks nodes as backed up, changes eviction eligibility via `dec_lock_ref` |
| `PREFETCH_FINALIZE` | Yes | Creates host tree nodes, changes `match_prefix` results |
| `PREFETCH_SKIP` | Yes | Marks prefetch as settled (no host nodes created) |
| `REVOKE` | Yes | Revokes a prefetch operation |

## Root Causes and Fixes

### Root Cause 1: Write Backup Barrier Missing on PP0

**Problem**: PP1 had a write-backup barrier in batch pick — it checks `pp_pending_host_tree_events` for `WRITE_BACKUP_COMMITTED` events that affect the request's tree path. If found, PP1 skips that request. But PP0 had no equivalent check. PP0 would pick requests that PP1 blocks on.

**Why it diverges**: After `init_next_round_input()` calls `match_prefix`, the prefix length depends on which nodes exist and their eviction state. A `WRITE_BACKUP_COMMITTED` event on PP0's outgoing queue means PP0 has already processed the write-back (changed node metadata), but PP1 hasn't seen it yet.

**Fix** (commit `9330729`): Added `has_outgoing_pp_write_backup_event_for_req()` to `HiRadixCache`. Extended the barrier condition in `scheduler.py` `get_new_batch_prefill()` to cover both ranks:

```python
# In scheduler.py get_new_batch_prefill()
_wb_barrier = False
if self.pp_group is not None and self.enable_hicache_storage:
    if (
        not self.pp_group.is_first_rank  # PP1: existing check
        and self.tree_cache.has_pending_pp_write_backup_event_for_req(req)
    ):
        _wb_barrier = True
    elif (
        self.pp_group.is_first_rank  # PP0: new check
        and self.tree_cache.has_outgoing_pp_write_backup_event_for_req(req)
    ):
        _wb_barrier = True
```

**Key files**:
- `python/sglang/srt/managers/scheduler.py` — barrier in `get_new_batch_prefill()`
- `python/sglang/srt/mem_cache/hiradix_cache.py` — `has_outgoing_pp_write_backup_event_for_req()`

### Root Cause 2: Write-Through Ack Consumption Divergence

**Problem**: PP0's `writing_check()` consumes write ack groups via `all_reduce(MIN)` across attention TP/CP groups. When an ack is consumed, it calls `dec_lock_ref()` which changes node eviction eligibility, potentially altering device tree structure. If PP0 has pending outgoing `WRITE_BACKUP_COMMITTED` events, it means PP1 hasn't yet seen the corresponding tree changes. If PP0 then consumes MORE acks, the divergence compounds.

**Fix** (commit `d054a12`): PP0 defers consuming write-through acks while `pp_outgoing_host_tree_events` is non-empty. This bounds the commit rate to the event delivery rate:

```python
# In hiradix_cache.py writing_check()
if (
    self.pp_size > 1
    and self.pp_rank < self.pp_size - 1
    and self._pp_write_backup_replay_enabled()
    and self.pp_outgoing_host_tree_events  # still have unsent events
):
    return  # defer ack consumption
```

### Root Cause 3: Prefetch Finalize Event Not Synchronized

**Problem**: PP0's `check_prefetch_progress()` finalizes a prefetch, which creates host tree nodes and emits a `PREFETCH_FINALIZE` event to `pp_outgoing_host_tree_events`. But this event hasn't reached PP1 yet. When `init_next_round_input()` calls `match_prefix`, PP0 sees newly created host nodes (longer prefix match) while PP1 still sees only the root (short prefix). Example: PP0 prefix=13056 vs PP1 prefix=64.

**Fix** (uncommitted, working tree): Added `has_outgoing_pp_prefetch_settle_event_for_req()` and a barrier in `scheduler.py` between `check_prefetch_progress()` and `init_next_round_input()`:

```python
# In scheduler.py get_new_batch_prefill(), after check_prefetch_progress
if (
    self.enable_hicache_storage
    and self.pp_group is not None
    and self.pp_group.is_first_rank
    and self.tree_cache.has_outgoing_pp_prefetch_settle_event_for_req(req.rid)
):
    break  # wait until event is delivered to PP1
```

## Debug Methodology

### Key Log Patterns

Use `grep` with these patterns to diagnose batch pick divergence:

```bash
# Batch pick results (what each rank decided to run)
grep "batch_pick" <logfile>

# Batch shape mismatch (the smoking gun for NCCL hangs)
grep "batch_shape_snapshot" <logfile>

# Head-of-queue empty pick reasons
grep "head_empty_pick" <logfile>

# Prefetch progress tracing
grep "PPPrefetchTrace" <logfile>

# HiCache prefetch thread activity
grep "HiCachePrefetchThread" <logfile>

# Write-backup replay events
grep "PPHiCacheSync" <logfile>
```

### Enabling Verbose HiCache Logs

Set `SGLANG_DEBUG_HICACHE_VERBOSE=1` to disable the `_HiCacheDebugFilter` and `_PPSchedulerDebugFilter` which suppress `[HiCache*]`, `[PPReqPhase]`, and `[PPHiCacheSync]` messages.

### Diagnostic Approach

1. **Identify the hang**: Look for NCCL timeout in logs. Note which iteration/step it occurs.
2. **Compare batch picks**: Find `batch_pick` or `batch_shape_snapshot` logs for both PP0 and PP1 at the same iteration. Check if request sets or prefix lengths differ.
3. **Trace the divergent request**: Find the request that appears in one rank's batch but not the other. Check `head_empty_pick` logs to see why it was skipped.
4. **Check event queues**: Look at `pp_outgoing_host_tree_events` size and `pp_pending_host_tree_events` drain rate. A growing outgoing queue on PP0 means events aren't being delivered fast enough.
5. **Check tree state**: Use `get_tree_shape_snapshot()` to compare device tree node counts, evicted counts, and backup counts between ranks.

## Design Principle: Route B (Event Sync)

The chosen approach is **Route B** — synchronize tree-mutating events between PP0 and PP1 to keep both trees consistent. The alternative (Route A: "settled frontier" where PP1 tells PP0 what to pick) was rejected because:

1. It requires an ack mechanism (PP1 → PP0 communication), adding latency.
2. It couples batch pick logic to cross-rank communication timing.
3. Route B leverages the natural event loop asymmetry: PP0 picks before sending, PP1 picks after receiving. Barriers are local checks on queue state, not network round-trips.

## Invariants to Maintain

1. **PP0 must not pick a request if it has outgoing events that affect that request's tree path.** The event hasn't reached PP1, so PP1's tree state is stale for that request.
2. **PP0 must not consume more write-through acks while outgoing events are pending.** This prevents compounding divergence.
3. **PP1 must replay all pending events before batch pick.** This is already enforced by `replay_pp_host_tree_events()` in `get_new_batch_prefill()`.
4. **Event types that create/modify tree nodes (WRITE_BACKUP_COMMITTED, PREFETCH_FINALIZE) are the primary divergence sources.** PREFETCH_SKIP and REVOKE also need barriers because they affect `check_prefetch_progress()` results.

## Related Files

| File | Role |
|------|------|
| `python/sglang/srt/managers/scheduler.py` | Batch pick logic, barrier checks in `get_new_batch_prefill()` |
| `python/sglang/srt/mem_cache/hiradix_cache.py` | HiRadixCache: event queues, barrier helpers, `writing_check()`, `check_prefetch_progress()` |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | PP event loop (`event_loop_pp`), event transport (`_pp_build_req_payload`, `_pp_apply_hicache_sync_before_batch`) |
| `python/sglang/srt/managers/cache_controller.py` | `HiCacheController`: write-back/prefetch thread management |
| `python/sglang/srt/disaggregation/prefill.py` | PD prefill bootstrap integration with PP |
