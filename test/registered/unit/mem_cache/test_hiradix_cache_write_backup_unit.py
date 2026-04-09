import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache, PPHostTreeEvent
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestHiRadixCacheWriteBackupBarrier(unittest.TestCase):
    def setUp(self):
        TreeNode.counter = 0
        self.cache = object.__new__(HiRadixCache)
        self.cache.root_node = TreeNode()
        self.cache.root_node.key = None
        self.cache.root_node.parent = None

    def _make_node(
        self,
        key_len: int,
        last_hash: str,
        *,
        extra_key: str | None = None,
        parent: TreeNode | None = None,
    ) -> TreeNode:
        node = TreeNode()
        node.key = RadixKey([0] * key_len, extra_key=extra_key)
        node.hash_value = [last_hash]
        node.parent = parent if parent is not None else self.cache.root_node
        return node

    def _make_write_backup_event(
        self,
        *,
        node_key_len: int,
        node_last_hash: str,
        parent_key_len: int = 0,
        parent_last_hash: str | None = None,
        extra_key: str | None = None,
        parent_extra_key: str | None = None,
    ) -> PPHostTreeEvent:
        return PPHostTreeEvent(
            seq=11,
            kind="WRITE_BACKUP_COMMITTED",
            node_ids=[1],
            node_key_lens=[node_key_len],
            node_last_hashes=[node_last_hash],
            node_extra_keys=[extra_key],
            node_parent_key_lens=[parent_key_len],
            node_parent_last_hashes=[parent_last_hash],
            node_parent_extra_keys=[parent_extra_key],
        )

    def test_host_hit_zero_ignores_stale_host_boundary(self):
        visible_device = self._make_node(64, "device-visible")
        stale_host = self._make_node(64, "stale-host")
        req = SimpleNamespace(
            host_hit_length=0,
            last_node=visible_device,
            last_host_node=stale_host,
        )
        event = self._make_write_backup_event(
            node_key_len=64,
            node_last_hash="stale-host",
        )

        self.assertFalse(self.cache._write_backup_event_affects_req(event, req))

    def test_host_hit_zero_still_matches_visible_device_boundary(self):
        visible_device = self._make_node(64, "device-visible")
        req = SimpleNamespace(
            host_hit_length=0,
            last_node=visible_device,
            last_host_node=self._make_node(64, "stale-host"),
        )
        event = self._make_write_backup_event(
            node_key_len=64,
            node_last_hash="device-visible",
        )

        self.assertTrue(self.cache._write_backup_event_affects_req(event, req))

    def test_host_hit_positive_matches_visible_host_boundary(self):
        visible_device = self._make_node(64, "device-visible")
        visible_host = self._make_node(320, "host-visible")
        req = SimpleNamespace(
            host_hit_length=2176,
            last_node=visible_device,
            last_host_node=visible_host,
        )
        event = self._make_write_backup_event(
            node_key_len=320,
            node_last_hash="host-visible",
        )

        self.assertTrue(self.cache._write_backup_event_affects_req(event, req))
