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

### Root Cause 4: L3 Prefetch Completion Timing Divergence

**Problem**: PP0 completes an L3 prefetch for a request (e.g., `9ec`) while it's still in bootstrap queue. The PREFETCH_FINALIZE event is emitted and delivered to PP1. However, PP1's own L3 prefetch for the same request hasn't even started (the request was in bootstrap on PP1). By the time both ranks release the request from bootstrap via consensus, PP0's `match_prefix` returns prefix=384 (64 device + 320 L3), while PP1's returns prefix=64 (no L3 data yet).

This causes `PrefillAdder` on PP0 to compute `new_tokens=45` for the request (accepted) while PP1 computes `new_tokens=365` (rejected due to token budget), resulting in different batch sizes (PP0: 6 reqs, PP1: 4 reqs) → shape crash in the forward pass.

**Key insight**: L3 cache hits are **deterministic** — both PP0 and PP1 compute the same token hashes, so if PP0 gets an L3 hit, PP1 will too. The only difference is *when* the KV download finishes. PP1 just needs to wait for its own download to complete.

**Why Root Cause 3 fix was insufficient**: The `has_outgoing_pp_prefetch_settle_event_for_req` barrier only checks the outgoing queue. By the time PP0 picks the request (seconds later), the event has already been consumed and sent. The barrier returns false.

**Why a watermark/round-counting approach was rejected**: An earlier iteration tracked `_pp_consume_seq` and barriered PP0 until N rounds passed. This used a magic number with no semantic guarantee — if L3 latency spikes, any fixed round count is wrong.

**Fix** (commit `0e5a5a8`): PP1 (follow rank) synchronous wait in `check_prefetch_progress()`. When `can_terminate_prefetch(operation)` returns False but the L3 hit is confirmed (`len(operation.hash_value) > 0`), PP1 spin-waits instead of returning False (which would break out and retry next iteration). Since the hit is deterministic, waiting guarantees PP1 gets the same prefix as PP0 in the same batch-pick iteration.

```python
# In hiradix_cache.py check_prefetch_progress()
if not self.can_terminate_prefetch(operation):
    # PP follow rank with confirmed L3 hit: synchronously wait for
    # the KV download to finish instead of returning False (which
    # would cause a break and retry next iteration).  The hit is
    # deterministic (same token hashes), so PP0 already has the same
    # data.  Waiting here makes PP1's prefix converge with PP0's in
    # the same batch-pick iteration — no ack or watermark needed.
    if (
        self.pp_size > 1
        and self.pp_rank > 0
        and len(operation.hash_value) > 0
        and not operation.is_terminated()
    ):
        while not self.can_terminate_prefetch(operation):
            time.sleep(0.001)
        # Fall through to _finalize_prefetch_progress below.
    else:
        return False
```

**Why this is safe**: `can_terminate_prefetch` already has TP-level `all_reduce` (across TP workers within the same PP rank) but NOT PP-level. The spin-loop calls `can_terminate_prefetch` with `time.sleep(0.001)`, which is safe because all TP workers on the same PP rank enter the spin together. The wait is bounded by L3 download time (typically milliseconds to low seconds).

**Zero transport changes** — purely local behavior on PP1. No impact on startup, warmup, non-PP, or non-storage modes.

### Root Cause 5: Micro-Batch Phase Misalignment from Asymmetric Empty Batches

**Problem**: In the disagg prefill PP event loop, PP0's sends (req/bootstrap/transfer/proxy) for mb_id=X are received by PP1 at mb_id=X+1 (1-step pipeline delay). The proxy tensor send is **conditional** on `cur_batch is not None`. When PP0's write-backup barrier (Root Cause 1) or prefetch barrier (Root Cause 3) causes an **empty batch** on PP0 at mb_id=X, PP0 skips the proxy send. PP1, however, may pick a non-empty batch at mb_id=X+1 (because PP1's tree state already had the events replayed). PP1 tries to recv proxy but PP0 never sent it — **permanent phase offset deadlock**.

**Why the existing barriers are insufficient**: The barriers in Root Causes 1-3 correctly prevent batch *content* divergence (same request, different prefix). But they create batch *presence* divergence (PP0 empty, PP1 non-empty). The 1-step pipeline offset means PP0's empty batch shifts all subsequent mb_ids by 1, and the proxy send/recv pairing is permanently broken.

**Symptom**: PP0 and PP1 batch_pick logs show identical request content but different mb_ids (e.g., PP0 picks `e52` on mb=0, PP1 picks `e52` on mb=1). PP0's batch_pick sequence has an extra empty batch (`batch=[]`) that PP1 doesn't have. PP0 has one more batch_pick iteration than PP1.

**Fix**: PP0 includes a `has_batch` flag in the req payload (`_PP_REQ_PAYLOAD_V2`). PP1 receives this flag in `recv_requests()` and saves it as `_pp_prev_stage_had_batch`. Before PP1's batch pick, if the flag is `False`, PP1 forces `batch=None` (skipping `get_new_batch_prefill()`). This keeps proxy tensor send/recv paired across the pipeline.

```python
# In event_loop_pp_disagg_prefill, before batch pick on PP1:
_pp_force_empty = (
    self.pp_group.is_last_rank
    and getattr(self, "_pp_prev_stage_had_batch", None) is False
)
if _pp_force_empty:
    batch = None
else:
    batch = self.get_new_batch_prefill()

# PP0 sends the flag in the req payload:
self._pp_build_req_payload(recv_reqs, has_batch=self.cur_batch is not None)
```

**Key files**:
- `python/sglang/srt/managers/scheduler_pp_mixin.py` — `_pp_pack_req_payload` (V2 format), `event_loop_pp_disagg_prefill` (force-empty gate)
- `python/sglang/srt/managers/scheduler.py` — `recv_requests()` (extract and broadcast `has_batch` flag)

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
