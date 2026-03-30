import bisect
import logging
import os
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import pygd2fs
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageHandler,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.utils.eventloop import EventLoop, EventType

logger = logging.getLogger(__name__)


class GD2FSIOPolicy(Enum):
    DIRECT = "direct"
    BOUNCE = "bounce"


class GD2FSDataPlaneKind(Enum):
    TCP = "tcp"
    RDMA = "rdma"


def get_protocol(uri: str) -> str:
    if not isinstance(uri, str) or "://" not in uri:
        raise ValueError("Invalid URI: missing '://'")
    return urllib.parse.urlparse(uri).scheme.lower()


@dataclass
class ReqContext:
    is_load: bool
    policy: GD2FSIOPolicy
    req: Optional[pygd2fs.Request]
    handler: HiCacheStorageHandler
    total_length: int
    mem_indices: torch.Tensor
    host_data: Optional[torch.Tensor] = None
    iomem: Optional[pygd2fs.IOMEM] = None
    mem_pool: Optional[KVCache | HostKVCache] = None


class GD2FSStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        if storage_config is None:
            raise ValueError("not found GD2FS storage configuration")

        self.tp_rank = storage_config.tp_rank
        self.is_mla_model = storage_config.is_mla_model

        extra_config = getattr(storage_config, "extra_config", None)
        if extra_config is None:
            raise ValueError("not found GD2FS configuration")

        self.cpaddr = extra_config.get("cpaddr", None)
        if self.cpaddr is None:
            raise ValueError("not found GD2FS cpaddr")

        self.dpaddr = extra_config.get("dpaddr", None)
        if self.dpaddr is None:
            raise ValueError("not found GD2FS dpaddr")

        dpkind = get_protocol(self.dpaddr)
        self.dpkind = GD2FSDataPlaneKind(dpkind)

        self.cluster = extra_config.get("cluster", None)
        if self.cluster is None:
            raise ValueError("not found GD2FS cluster")

        self.iothreads = extra_config.get("iothreads", 1)
        self.memthreads = extra_config.get("memthreads", 1)
        self.streams = extra_config.get("streams", 1)

        self.client = pygd2fs.Client(
            self.cpaddr,
            self.dpaddr,
            self.cluster,
            iothreads=self.iothreads,
            memthreads=self.memthreads,
            streams=self.streams,
        )
        if self.client is None:
            raise RuntimeError("cannot connect to GD2FS cluster")

        self.fd = self.client.GetFd()
        self.req_handlers = {}

        self.mem_regions = []

        self.load_policy = self.get_io_policy("GD2FS_LOAD_POLICY")
        self.save_policy = self.get_io_policy("GD2FS_SAVE_POLICY")
        self.check_io_policy()

    def get_io_policy(self, envkey: str) -> GD2FSIOPolicy:
        policy = os.getenv(envkey, None)
        if policy is None:
            if self.dpkind == GD2FSDataPlaneKind.TCP:
                policy = "bounce"
            elif self.dpkind == GD2FSDataPlaneKind.RDMA:
                policy = "direct"
        return GD2FSIOPolicy(policy.lower())

    def check_io_policy(self):
        if self.dpkind == GD2FSDataPlaneKind.TCP:
            if self.load_policy == GD2FSIOPolicy.DIRECT:
                raise ValueError(
                    "GD2FS load policy direct is not supported for tcp data plane"
                )

            if self.save_policy == GD2FSIOPolicy.DIRECT:
                raise ValueError(
                    "GD2FS save policy direct is not supported for tcp data plane"
                )

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[Optional[dict], str]:
        if not uri.startswith("gd2fs://"):
            return (None, uri)

        parsed = urllib.parse.urlparse(uri)

        addr_list = [a.strip() for a in parsed.netloc.split(",") if a.strip()]
        cpaddr = ",".join(f"gd2fs://{a}" for a in addr_list)

        dpaddr = os.getenv("GD2FS_DPADDR", "tcp://127.0.0.1")
        iothreads = int(os.getenv("GD2FS_IOTHREADS", 1))
        memthreads = int(os.getenv("GD2FS_MEMTHREADS", 1))
        streams = int(os.getenv("GD2FS_STREAMS", 1))

        query = {
            k: v[0] if v else "" for k, v in urllib.parse.parse_qs(parsed.query).items()
        }

        return {
            "cpaddr": cpaddr,
            "dpaddr": dpaddr,
            "iothreads": iothreads,
            "memthreads": memthreads,
            "streams": streams,
            **query,
        }, parsed.path

    def register_mem_pool_device(self, mem_pool_device: KVCache) -> None:
        self.mem_pool_device = mem_pool_device
        kv_data_ptrs, kv_data_lens, _ = self.mem_pool_device.get_contiguous_buf_infos()

        for kv_data_ptr, kv_data_len in zip(kv_data_ptrs, kv_data_lens):
            iomem = self.client.RegIOMEM(kv_data_ptr, kv_data_len)
            if iomem is None:
                raise RuntimeError(
                    f"cannot register memory [{kv_data_ptr}, {kv_data_ptr+kv_data_len}] to GD2FS client"
                )
            end_addr = kv_data_ptr + kv_data_len
            bisect.insort(self.mem_regions, (kv_data_ptr, end_addr, iomem))

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        self.mem_pool_host = mem_pool_host
        kv_buffer = self.mem_pool_host.kv_buffer
        iomem = self.client.RegIOMEM(
            kv_buffer.data_ptr(), kv_buffer.numel() * kv_buffer.element_size()
        )
        if iomem is None:
            raise RuntimeError("cannot register memory to GD2FS client")

        bisect.insort(
            self.mem_regions,
            (
                kv_buffer.data_ptr(),
                kv_buffer.data_ptr() + kv_buffer.numel() * kv_buffer.element_size(),
                iomem,
            ),
        )

    def _get_iomem_by_address(self, addr: int, length: int):
        idx = bisect.bisect_right(self.mem_regions, (addr, float("inf"))) - 1
        if idx < 0:
            raise RuntimeError(
                f"memory region [{addr}, {addr + length}] not registered in GD2FS storage"
            )

        start_addr, end_addr, iomem = self.mem_regions[idx]
        if not (start_addr <= addr <= end_addr - length + 1):
            raise RuntimeError(
                f"memory region [{addr}, {addr + length}] not registered in GD2FS storage fully"
            )

        return iomem

    def _wait_request(self, req: pygd2fs.Request) -> tuple[int, int]:
        if req is None:
            raise RuntimeError("GD2FS request is None")

        reqs = self.client.Wait(-1)
        if len(reqs) > 1:
            raise RuntimeError(f"wait {len(reqs)} requests, expect 1")

        if reqs[0] != req:
            raise RuntimeError(f"wait wrong request {reqs[0]}, expect {req}")

        return req.Status(), req.Value()

    def _calculate_total_length(self, meta: list[tuple[int, int]]) -> int:
        return sum(len for _, len in meta)

    def _create_sge_direct(self, meta: list[tuple[int, int]]) -> list[pygd2fs.SGE]:
        sgs = []
        for addr, size in meta:
            iomem = self._get_iomem_by_address(addr, size)
            sgs.append(pygd2fs.SGE(addr, size, iomem))
        return sgs

    def _create_sge_bounce(
        self, host_data: torch.Tensor
    ) -> tuple[list[pygd2fs.SGE], Any]:
        length = host_data.numel() * host_data.element_size()
        data_ptr = host_data.data_ptr()
        iomem = self.client.RegIOMEM(data_ptr, length)
        sgs = [pygd2fs.SGE(data_ptr, length, iomem)]

        return sgs, iomem

    def wait_all(self):
        while True:
            reqs = self.client.Wait(0)
            if not reqs:
                return

            for req in reqs:
                self.req_handlers[req]()
                del self.req_handlers[req]

    def _register_fd(self, evloop: EventLoop):
        def waitall(fd: int, events: EventType, data: Any):
            self.wait_all()

        if not evloop.fd_is_registered(self.fd, EventType.READ):
            evloop.register_fd(self.fd, EventType.READ, waitall)

    def _req_complete(self, ctx: ReqContext, exception: Optional[Exception] = None):
        req = ctx.req

        if exception is None:
            if req is None:
                exception = RuntimeError("GD2FS read error")
            elif req.Status() != 0:
                exception = RuntimeError(f"GD2FS request failed, {req}")
            elif req.Value() != ctx.total_length:
                exception = RuntimeError(
                    f"GD2FS request value {req.Value()} not equal to total length {ctx.total_length}"
                )

        if ctx.is_load and ctx.policy == GD2FSIOPolicy.BOUNCE and exception is None:
            assert ctx.host_data is not None and ctx.mem_pool is not None
            ctx.mem_pool.set_from_flat_data(ctx.mem_indices, ctx.host_data)

        if ctx.iomem is not None:
            self.client.DeregIOMEM(ctx.iomem)

        ctx.handler(exception)

    def _add_req_complete(self, req: pygd2fs.Request, ctx: ReqContext):
        self.req_handlers[req] = lambda ctx=ctx: self._req_complete(ctx)

    def load(
        self,
        evloop: EventLoop,
        filepath: str,
        offset: int,
        mem_pool: KVCache | HostKVCache,
        mem_indices: torch.Tensor,
        handler: HiCacheStorageHandler,
    ) -> None:
        if not mem_indices.numel():
            handler(None)
            return

        try:
            sgs: list[pygd2fs.SGE] = []
            iomem: Optional[pygd2fs.IOMEM] = None
            host_data: Optional[torch.Tensor] = None

            meta = mem_pool.get_buffer_meta(mem_indices)
            total_length = self._calculate_total_length(meta)

            if self.load_policy == GD2FSIOPolicy.DIRECT:
                sgs = self._create_sge_direct(meta)
            elif self.load_policy == GD2FSIOPolicy.BOUNCE:
                host_data = torch.empty(
                    int(total_length / mem_pool.dtype.itemsize),
                    dtype=mem_pool.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                sgs, iomem = self._create_sge_bounce(host_data)
        except Exception as e:
            handler(e)
            return

        ctx = ReqContext(
            is_load=True,
            policy=self.load_policy,
            req=None,
            handler=handler,
            total_length=total_length,
            mem_indices=mem_indices,
            host_data=host_data,
            iomem=iomem,
            mem_pool=mem_pool,
        )

        try:
            req = self.client.Read(filepath, offset, sgs, 0, "")
            if req is None:
                raise RuntimeError("GD2FS request cannot be created")

            ctx.req = req

            self._add_req_complete(req, ctx)
            self._register_fd(evloop)

        except Exception as e:
            self._req_complete(ctx, e)

    def save(
        self,
        evloop: EventLoop,
        filepath: str,
        offset: int,
        mem_pool_device: KVCache,
        mem_indices: torch.Tensor,
        handler: HiCacheStorageHandler,
    ) -> None:
        if not mem_indices.numel():
            handler(None)
            return

        if self.is_mla_model and self.tp_rank != 0:
            handler(None)
            return

        try:
            sgs = []
            iomem: Optional[pygd2fs.IOMEM] = None
            host_data: Optional[torch.Tensor] = None
            total_length = 0

            if self.save_policy == GD2FSIOPolicy.DIRECT:
                meta = mem_pool_device.get_buffer_meta(mem_indices)
                sgs = self._create_sge_direct(meta)
                total_length = self._calculate_total_length(meta)
            elif self.save_policy == GD2FSIOPolicy.BOUNCE:
                host_data = mem_pool_device.get_flat_data(mem_indices)
                sgs, iomem = self._create_sge_bounce(host_data)
                total_length = host_data.numel() * host_data.element_size()
        except Exception as e:
            handler(e)
            return

        ctx = ReqContext(
            is_load=False,
            policy=self.save_policy,
            req=None,
            handler=handler,
            total_length=total_length,
            mem_indices=mem_indices,
            host_data=host_data,
            iomem=iomem,
            mem_pool=None,
        )

        try:
            req = self.client.Write(filepath, offset, sgs, 0, "")
            if req is None:
                raise RuntimeError("GD2FS request cannot be created")

            ctx.req = req

            self._add_req_complete(req, ctx)
            self._register_fd(evloop)

        except Exception as e:
            self._req_complete(ctx, e)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError
