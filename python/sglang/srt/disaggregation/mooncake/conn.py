from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import socket
import struct
import threading
from functools import cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import zmq, socket
from aiohttp import web
import requests

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils import get_free_port, get_ip, get_local_ip_by_remote
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    src_groups = []
    dst_groups = []
    current_src = [src_indices[0]]
    current_dst = [dst_indices[0]]

    for i in range(1, len(src_indices)):
        src_contiguous = src_indices[i] == src_indices[i - 1] + 1
        dst_contiguous = dst_indices[i] == dst_indices[i - 1] + 1
        if src_contiguous and dst_contiguous:
            current_src.append(src_indices[i])
            current_dst.append(dst_indices[i])
        else:
            src_groups.append(current_src)
            dst_groups.append(current_dst)
            current_src = [src_indices[i]]
            current_dst = [dst_indices[i]]

    src_groups.append(current_src)
    dst_groups.append(current_dst)

    return src_groups, dst_groups


@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int64]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_ptrs: list[int]
    dst_aux_index: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            endpoint=msg[0].decode("ascii"),
            dst_port=int(msg[1].decode("ascii")),
            mooncake_session_id=msg[2].decode("ascii"),
            room=int(msg[3].decode("ascii")),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_kv_indices=np.frombuffer(msg[5], dtype=np.int64),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_aux_index=int(msg[7].decode("ascii")),
        )


class MooncakeKVManager(BaseKVManager):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()
    def __init__(self, args: KVArgs, disaggregation_mode: DisaggregationMode, server_args: ServerArgs):
        self.engine = MooncakeTransferEngine(args.ib_device)
        self.kv_args = args
        self.server_args = server_args
        self.disaggregation_mode = disaggregation_mode
        self.request_status: Dict[int, KVPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        self.rank_port = None

        # TODO: in decode node, this is no need to be called   in the future
        # For now, prefill mulit-node need use the same prefill addr registerd to bootstrap server
        self.prefill_addr = self.parse_prefill_addr(server_args)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_queue = queue.Queue()
            self.transfer_infos: Dict[int, TransferInfo] = {}
            self.start_prefill_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.start_decode_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def parse_prefill_addr(self, args: ServerArgs):
        port = args.port  # http_server_port
        dist_init_addr = args.dist_init_addr
        if dist_init_addr:
            ip_address = socket.gethostbyname(dist_init_addr.split(":")[0])
        else:
            ip_address = self.engine.get_localhost()
        return f"{ip_address}:{port}"

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    def register_zmq_info_to_bootstrap_server(self, key, value, bootstrap_addr: str = None):
        """
        register to the bootstrap server
        """
        bootstrap_url = self.parse_bootstrap_addr(self.server_args, bootstrap_addr)
        respo = requests.put(bootstrap_url, params={"key": key}, json=value)
        if respo.status_code != 200:
            raise Exception("error registering zmq info to meta data")

    def parse_bootstrap_addr(self, args: ServerArgs, bootstrap_addr: str = ""):
        """
        parse the bootstrap addr from the server args
        """
        scheme = "http://"

        if bootstrap_addr:
            # for now bootstrap_port is fixed to 8998
            return f"{scheme}{bootstrap_addr}/kv_route"

        else:
            port = args.disaggregation_bootstrap_port  # bootstrap_port
            dist_init_addr = args.dist_init_addr
            if dist_init_addr:
                ip_address = socket.gethostbyname(dist_init_addr.split(":")[0])
            else:
                ip_address = self.engine.get_localhost()
            return f"{scheme}{ip_address}:{port}/kv_route"


    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @cache
    def query_zmq_rank_addr(self, key, bootstrap_url):
        resp = requests.get(bootstrap_url, params={"key": key})
        if resp.status_code != 200:
            raise Exception("Cant query receiver rank port for key {}, resp status {}".format(key, resp.status_code))
        ip, port = resp.json()['zmq_ip'], resp.json()['zmq_port']
        return ip, port

    @cache
    def get_free_zmq_port(self, key):
        return get_free_port()

    @cache
    def get_prefill_register_key(self, prefill_addr: str):
        key = f'{prefill_addr}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}'
        return key

    @cache
    def get_decode_register_key(self, session_id):
        key = f'{session_id}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}'
        return key

    @cache
    def get_pd_meta_key(self):
        """
        """
        return f"{self.engine.session_id}_{self.disaggregation_mode}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}"

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        for layer_id in range(num_layers):
            src_ptr = self.kv_args.kv_data_ptrs[layer_id]
            dst_ptr = dst_kv_ptrs[layer_id]
            item_len = self.kv_args.kv_item_lens[layer_id]

            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)

                # TODO: make async later
                status = self.engine.transfer_sync(
                    mooncake_session_id, src_addr, dst_addr, length
                )
                if status != 0:
                    return status

        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        aux_item_len = self.kv_args.aux_item_lens[0]
        prefill_aux_addr = (
            self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        )
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        # TODO: mooncake transfer engine can do async transfer. Do async later
        # Not sure about the amount of aux data, maybe transfer it by zmq is more effective
        status = self.engine.transfer_sync(
            mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
        )
        return status
    def sync_status_to_decode_endpoint(self, remote: str, dst_port: int, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        sock, lock = self._connect("tcp://" + remote + ":" + str(dst_port))
        with lock:
            sock.send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )
    def start_prefill_thread(self):
        self.rank_port = self.get_free_zmq_port(self.get_pd_meta_key())
        # should register to the bootstrap server, now to metadata server
        self.register_zmq_info_to_bootstrap_server(
            self.get_prefill_register_key(self.prefill_addr),
            {"zmq_port": self.rank_port, "zmq_ip": self.engine.get_localhost()}
        )
        self.server_socket.bind("tcp://*:" + str(self.rank_port))

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    continue
                room = int(room)
                self.transfer_infos[room] = TransferInfo.from_zmq(waiting_req_bytes)

                # NOTE: after bootstrapping we can mark the req as waiting for input
                self.request_status[room] = KVPoll.WaitingForInput

        def transfer_thread():
            # TODO: Shall we use KVPoll.Transferring state?
            while True:
                try:
                    kv_chunk: TransferKVChunk = self.transfer_queue.get(timeout=0.01)
                    req = self.transfer_infos[kv_chunk.room]
                    chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]
                    assert len(chunked_dst_kv_indice) == len(
                        kv_chunk.prefill_kv_indices
                    )

                    ret = self.send_kvcache(
                        req.mooncake_session_id,
                        kv_chunk.prefill_kv_indices,
                        req.dst_kv_ptrs,
                        chunked_dst_kv_indice,
                    )
                    if ret != 0:
                        self.request_status[kv_chunk.room] = KVPoll.Failed
                        self.sync_status_to_decode_endpoint(
                            req.endpoint, req.dst_port, req.room
                        )
                        continue

                    if kv_chunk.is_last:
                        # Only the last chunk we need to send the aux data
                        ret = self.send_aux(
                            req.mooncake_session_id,
                            kv_chunk.prefill_aux_index,
                            req.dst_aux_ptrs,
                            req.dst_aux_index,
                        )
                        self.request_status[req.room] = (
                            KVPoll.Success if ret == 0 else KVPoll.Failed
                        )
                        self.sync_status_to_decode_endpoint(
                            req.endpoint, req.dst_port, req.room
                        )
                        self.transfer_infos.pop(req.room)

                except queue.Empty:
                    continue

        threading.Thread(target=bootstrap_thread).start()
        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        self.rank_port = self.get_free_zmq_port(self.get_pd_meta_key())
        self.server_socket.bind("tcp://*:" + str(self.rank_port))

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        threading.Thread(target=decode_thread).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        self.transfer_queue.put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )
        self.request_status[bootstrap_room] = KVPoll.WaitingForInput

    def check_status(self, bootstrap_room: int):
        # TOOD: do we really need the poll()?

        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: The prefill engine could recv bootstrapping first
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
            )

    def get_localhost(self):
        return self.engine.get_localhost()

    def get_session_id(self):
        return self.engine.get_session_id()

class MooncakeKVSender(BaseKVSender):

    def __init__(
        self, mgr: MooncakeKVManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.session_id = self.kv_mgr.get_session_id()

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
    ):
        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class MooncakeKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()
    def __init__(
        self,
        mgr: MooncakeKVManager,
        prefill_addr: str,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.prefill_addr = prefill_addr
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.bootstrap_url = self.kv_mgr.parse_bootstrap_addr(mgr.server_args, bootstrap_addr)
        prefill_ip, sender_rank_port = self.kv_mgr.query_zmq_rank_addr(
            self.kv_mgr.get_prefill_register_key(prefill_addr), self.bootstrap_url)

        logger.debug(f"KVReceiver init: prefill_addr={prefill_addr}, sender_rank_port={sender_rank_port}")
        self.prefill_server_url = (
            prefill_ip
            + ":"
            + str(sender_rank_port)
        )
        #self.register_to_bootstrap(self.prefill_server_url)

        self.decode_ip = self.kv_mgr.get_localhost()
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(bootstrap_room, KVPoll.WaitingForInput)

    def _get_prefill_addr_from_bootstrap(self, tp_rank: int):
        """Fetch the prefill server port corresponding to tp_rank from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/kv_route?tp_rank={tp_rank}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_addr = response.json()
                return prefill_addr
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    @cache
    def register_to_bootstrap(self, prefill_addr: str) -> None:
        logger.debug(f"KVReceiver Registering prefill_addr={prefill_addr}")
        # should register to the bootstrap server, now to metadata server
        self.kv_mgr.register_zmq_info_to_bootstrap_server(
            self.kv_mgr.get_decode_register_key(self.kv_mgr.engine.get_session_id()),
            {"zmq_port": self.kv_mgr.rank_port, "zmq_ip": self.kv_mgr.engine.get_localhost()}, bootstrap_addr=self.bootstrap_addr)

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]


    def init(
        self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None
    ):
        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )
        sock, lock = self._connect("tcp://" + self.prefill_server_url)
        with lock:
            sock.send_multipart(
            [
                self.decode_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
                str(self.bootstrap_room).encode("ascii"),
                packed_kv_data_ptrs,
                kv_indices.tobytes(),
                packed_aux_data_ptrs,
                str(aux_index).encode("ascii"),
            ]
        )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class MooncakeKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.kv_route_store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        # will deprecate in the future
        self.app.router.add_route("*", "/metadata", self._handle_metadata)

        # only route for bootstrap server
        self.app.router.add_route("*", "/kv_route", self._handle_kv_route)

    async def _handle_metadata(self, request: web.Request):
        key = request.query.get("key", "")

        if request.method == "GET":
            return await self._handle_metadata_get(key)
        elif request.method == "PUT":
            return await self._handle_metadata_put(key, request)
        elif request.method == "DELETE":
            return await self._handle_metadata_delete(key)
        return web.Response(
            text="Method not allowed", status=405, content_type="application/json"
        )

    async def _handle_metadata_get(self, key):
        async with self.lock:
            value = self.store.get(key)
        if value is None:
            return web.Response(
                text="metadata not found", status=404, content_type="application/json"
            )
        return web.Response(body=value, status=200, content_type="application/json")

    async def _handle_metadata_put(self, key, request):
        data = await request.read()
        async with self.lock:
            self.store[key] = data
        return web.Response(
            text="metadata updated", status=200, content_type="application/json"
        )

    async def _handle_metadata_delete(self, key):
        async with self.lock:
            if key not in self.store:
                return web.Response(
                    text="metadata not found",
                    status=404,
                    content_type="application/json",
                )
            del self.store[key]
        return web.Response(
            text="metadata deleted", status=200, content_type="application/json"
        )

    async def _handle_kv_route(self, request: web.Request):
        key = request.query.get("key", "")

        if request.method == "GET":
            return await self._handle_kv_route_get(key)
        elif request.method == "PUT":
            return await self._handle_kv_route_put(key, request)
        elif request.method == "DELETE":
            return await self._handle_metadata_delete(key)
        return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_kv_route_put(self, key, request: web.Request):
        data = await request.read()
        async with self.lock:
            self.kv_route_store[key] = data
        return web.Response(
            text="kv route info updated", status=200, content_type="application/json"
            )

    async def _handle_kv_route_get(self, key):
        async with self.lock:
            value = self.kv_route_store.get(key)
        if value is None:
            return web.Response(
                text="metadata not found", status=404, content_type="application/json"
            )
        return web.Response(body=value, status=200, content_type="application/json")

    async def _handle_kv_route_delete(self, key):
        async with self.lock:
            if key not in self.kv_route_store:
                return web.Response(
                    text="metadata not found",
                    status=404,
                    content_type="application/json",
                )
            del self.kv_route_store[key]
        return web.Response(
            text="metadata deleted", status=200, content_type="application/json"
        )

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._runner = web.AppRunner(self.app)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
