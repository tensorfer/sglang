from __future__ import annotations

import json
import logging
import os
import queue
import threading
from abc import abstractmethod
from dataclasses import asdict, dataclass, replace
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import KVCache, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import current_platform, parse_connector_type
from sglang.srt.utils.eventloop import EventLoop, EventType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExKVCacheSegment:
    """
    Represents a segment of cached tokens and their corresponding KV cache location.

    Attributes:
        token_start: The starting token ID within the logical token sequence.
        token_length: The number of tokens in this segment.
        kv_uri: A URI string identifying the storage backend and path for the KV cache.
        kv_start: The starting position in the physical KV cache storage.
        kv_length: The length of the KV cache data in the storage (optional).
                   If None, it's typically inferred from token_length * kv_length_per_token.
    """

    token_start: int
    token_length: int
    kv_uri: str
    kv_start: int
    kv_length: Optional[int] = None

    def __post_init__(self):
        if self.token_start < 0:
            raise ValueError(
                f"token_start must be non-negative, got {self.token_start}"
            )
        if self.token_length <= 0:
            raise ValueError(f"token_length must be positive, got {self.token_length}")
        if self.kv_start < 0:
            raise ValueError(f"kv_start must be non-negative, got {self.kv_start}")
        if self.kv_length is not None and self.kv_length <= 0:
            raise ValueError(f"kv_length must be positive if set, got {self.kv_length}")

    @classmethod
    def from_dict(cls, d: Dict) -> ExKVCacheSegment:
        """Creates an instance from a dictionary."""
        required_fields = ["token_start", "token_length", "kv_uri", "kv_start"]
        missing_fields = [f for f in required_fields if f not in d]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in ExKVCacheSegment dictionary: {missing_fields}. "
                f"Required fields: {required_fields}"
            )
        return cls(
            token_start=d["token_start"],
            token_length=d["token_length"],
            kv_uri=d["kv_uri"],
            kv_start=d["kv_start"],
            kv_length=d.get("kv_length", None),
        )

    def to_dict(self) -> Dict:
        """Converts the instance to a dictionary."""
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length


class ExKVCache:
    """
    Manages a collection of ExKVCacheSegment instances representing a session's KV cache state.

    This class handles truncating, binding token IDs/memory indices,
    and interacting with storage backends for prefetching and backing up cache data.
    """

    def __init__(
        self,
        segments: Optional[Union[List[ExKVCacheSegment], List[Dict]]] = None,
        offset: int = 0,
        token_ids: Optional[List[int]] = None,
        mem_indices: Optional[torch.Tensor] = None,
        storage_config: Optional[HiCacheStorageConfig] = None,
        kv_length_per_token: Optional[int] = None,
    ):
        self._offset: int = offset
        self._token_ids: Optional[List[int]] = token_ids
        self._mem_indices: Optional[torch.Tensor] = mem_indices
        self._storage_config: Optional[HiCacheStorageConfig] = storage_config
        self._kv_length_per_token: Optional[int] = kv_length_per_token

        if segments is None:
            self._segments: Tuple[ExKVCacheSegment, ...] = ()
            return

        processed: List[ExKVCacheSegment] = []
        for seg in segments:
            if isinstance(seg, dict):
                processed.append(ExKVCacheSegment.from_dict(seg))
            else:
                processed.append(seg)

        if not processed:
            self._segments = ()
            return

        sorted_segments = sorted(processed, key=lambda s: s.token_start)

        prev_end = sorted_segments[0].token_start
        for i, seg in enumerate(sorted_segments):
            if seg.token_start != prev_end:
                raise ValueError(
                    f"Segments are not contiguous at index {i}: "
                    f"...[{sorted_segments[i-1].token_start}, "
                    f"{sorted_segments[i-1].token_end}], "
                    f"[{seg.token_start}, {seg.token_end}]..."
                    if i > 0
                    else ""
                )
            prev_end = seg.token_end

        _segments = ()
        if self._storage_config is not None and self._kv_length_per_token is not None:
            sorted_kv_segments: List[ExKVCacheSegment] = []
            for seg in sorted_segments:
                kv_length = seg.token_length * self._kv_length_per_token
                kv_start = seg.kv_start
                if not self._storage_config.is_mla_model:
                    kv_start += self._storage_config.tp_rank * kv_length
                sorted_kv_segments.append(
                    ExKVCacheSegment(
                        token_start=seg.token_start,
                        token_length=seg.token_length,
                        kv_uri=seg.kv_uri,
                        kv_start=kv_start,
                        kv_length=kv_length,
                    )
                )
            _segments = tuple(sorted_kv_segments)
        else:
            _segments = tuple(sorted_segments)

        self._segments = _segments

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> ExKVCacheSegment:
        return self._segments[index]

    def __iter__(self) -> Iterator[ExKVCacheSegment]:
        return iter(self._segments)

    def __repr__(self) -> str:
        return f"ExKVCache(segments={list(self._segments)}, _offset={self._offset})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExKVCache):
            return False
        return self._segments == other._segments and self._offset == other._offset

    def __hash__(self) -> int:
        return hash((self._segments, self._offset))

    @property
    def token_start(self) -> int:
        """The logical start token ID of the entire cache."""
        return self._segments[0].token_start + self._offset if self._segments else 0

    @property
    def token_end(self) -> int:
        """The logical end token ID (exclusive) of the entire cache."""
        return self._segments[-1].token_end if self._segments else 0

    @property
    def token_length(self) -> int:
        """The total logical token length covered by the cache."""
        return self.token_end - self.token_start

    @property
    def token_range(self) -> Optional[Tuple[int, int]]:
        """Returns the (start, end) logical token range, or None if empty."""
        if not self._segments:
            return None
        return (self.token_start, self.token_end)

    @property
    def real_token_start(self) -> int:
        """The real start token ID of the entire cache."""
        return self._segments[0].token_start if self._segments else 0

    @property
    def real_token_end(self) -> int:
        """The real end token ID (exclusive) of the entire cache."""
        return self.token_end

    @property
    def real_token_length(self) -> int:
        """The real token length including the internal offset."""
        return self.real_token_end - self.real_token_start

    def to_dicts(self) -> List[Dict]:
        """Serializes the segments into a list of dictionaries."""
        return [seg.to_dict() for seg in self._segments]

    @classmethod
    def _from_validated_segments_and_state(
        cls,
        segments: Tuple[ExKVCacheSegment, ...],
        offset: int = 0,
        token_ids: Optional[List[int]] = None,
        mem_indices: Optional[torch.Tensor] = None,
        storage_config: Optional[HiCacheStorageConfig] = None,
        kv_length_per_token: Optional[int] = None,
    ) -> ExKVCache:
        """Internal constructor for creating instances with pre-validated state."""
        instance = object.__new__(cls)
        instance._segments = segments
        instance._offset = offset
        instance._token_ids = token_ids
        instance._mem_indices = mem_indices
        instance._storage_config = storage_config
        instance._kv_length_per_token = kv_length_per_token
        return instance

    def bind_token_ids(self, token_ids: List[int]) -> ExKVCache:
        """
        Returns a new ExKVCache instance with token_ids bound.

        Raises:
            ValueError: If the length of token_ids doesn't match the token_length.
        """
        if len(token_ids) != self.token_length:
            raise ValueError(
                f"Token IDs length {len(token_ids)} does not match token length {self.token_length}"
            )

        return ExKVCache._from_validated_segments_and_state(
            segments=self._segments,
            offset=self._offset,
            token_ids=token_ids,
            mem_indices=self._mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    def bind_mem_indices(self, mem_indices: torch.Tensor) -> ExKVCache:
        """
        Returns a new ExKVCache instance with mem_indices bound.

        Raises:
            ValueError: If the length of mem_indices doesn't match the real_token_length.
        """
        if len(mem_indices) != self.real_token_length:
            raise ValueError(
                f"Memory indices length {len(mem_indices)} does not match real token length {self.real_token_length}"
            )

        return ExKVCache._from_validated_segments_and_state(
            segments=self._segments,
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    @property
    def token_ids(self) -> List[int]:
        """Returns the bound token IDs.

        Raises:
            ValueError: If token_ids are not bound.
        """
        if self._token_ids is None:
            raise ValueError("token_ids are not bound")
        return self._token_ids

    @property
    def mem_indices(self) -> torch.Tensor:
        """Returns the valid bound memory indices.

        Raises:
            ValueError: If mem_indices are not bound.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")
        return self._mem_indices[self._offset :]

    def real_mem_indices_prefix(self, offset: int = 0) -> torch.Tensor:
        """Returns the prefix of the real bound memory indices.

        Raises:
            ValueError: If mem_indices are not bound.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")
        return self._mem_indices[: self._offset + offset]

    def _shift_real_token(self, offset: Optional[int] = None) -> ExKVCache:
        """
        Returns a new ExKVCache with all segment token_start/ends shifted.

        Args:
            offset: The amount to shift. If None, shifts so that token_start becomes 0.
        """
        if not self._segments:
            return ExKVCache()

        actual_offset = offset if offset is not None else -self.real_token_start

        new_segments = [
            ExKVCacheSegment(
                token_start=seg.token_start + actual_offset,
                token_length=seg.token_length,
                kv_uri=seg.kv_uri,
                kv_start=seg.kv_start,
                kv_length=seg.kv_length,
            )
            for seg in self._segments
        ]

        return ExKVCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=self._mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    def result(self) -> ExKVCache:
        """Returns a new ExKVCache with all TP ranks' segments."""
        if (
            self._storage_config is None
            or self._storage_config.tp_size == 1
            # Only TP0 needs to and is capable of returning a valid value.
            or self._storage_config.tp_rank > 0
        ):
            return self

        count = 1 if self._storage_config.is_mla_model else self._storage_config.tp_size

        new_segments = [
            ExKVCacheSegment(
                token_start=seg.token_start,
                token_length=seg.token_length,
                kv_uri=seg.kv_uri,
                kv_start=seg.kv_start,
                kv_length=seg.kv_length * count if seg.kv_length else 0,
            )
            for seg in self._segments
        ]

        return ExKVCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=self._mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    def truncate_prefix(self, offset: int) -> ExKVCache:
        """
        Returns a new ExKVCache truncated from the beginning up to 'offset'.

        Args:
            offset: The logical token offset up to which to truncate.

        Raises:
            ValueError: If offset is less than token_start.
        """
        if not self._segments:
            return ExKVCache()

        if offset < self.token_start:
            raise ValueError(
                f"offset ({offset}) must be greater than or equal to the start of the cache ({self.token_start})"
            )

        if offset >= self.token_end:
            return ExKVCache()

        new_segments = [seg for seg in self._segments if seg.token_end > offset]
        if not new_segments:
            return ExKVCache()

        new_offset = offset - new_segments[0].token_start

        new_token_ids = None
        if self._token_ids is not None:
            new_token_ids = self._token_ids[offset - self.token_start :]

        new_mem_indices = None
        if self._mem_indices is not None:
            new_real_offset = new_segments[0].token_start - self.real_token_start
            new_mem_indices = self._mem_indices[new_real_offset:]

        return ExKVCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=new_offset,
            token_ids=new_token_ids,
            mem_indices=new_mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    def truncate_suffix(self, offset: int) -> ExKVCache:
        """
        Returns a new ExKVCache truncated from 'offset' to the end.

        Args:
            offset: The logical token offset from which to truncate.
        """
        if not self._segments:
            return ExKVCache()

        if offset <= self.token_start:
            return ExKVCache()

        if offset >= self.token_end:
            new_segments = [
                ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=seg.kv_length,
                )
                for seg in self._segments
            ]
            return ExKVCache._from_validated_segments_and_state(
                segments=tuple(new_segments),
                offset=self._offset,
                token_ids=self._token_ids,
                mem_indices=self._mem_indices,
                storage_config=self._storage_config,
                kv_length_per_token=self._kv_length_per_token,
            )

        new_segments = []
        for seg in self._segments:
            if seg.token_start >= offset:
                break
            elif seg.token_end <= offset:
                new_seg = ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=seg.kv_length,
                )
                new_segments.append(new_seg)
            else:
                new_token_length = offset - seg.token_start
                new_kv_length = (
                    new_token_length * self._kv_length_per_token
                    if self._kv_length_per_token
                    else None
                )
                if new_token_length > 0:
                    new_seg = ExKVCacheSegment(
                        token_start=seg.token_start,
                        token_length=new_token_length,
                        kv_uri=seg.kv_uri,
                        kv_start=seg.kv_start,
                        kv_length=new_kv_length,
                    )
                    new_segments.append(new_seg)
                break

        if not new_segments:
            return ExKVCache()

        new_token_ids = None
        if self._token_ids is not None:
            new_token_ids = self._token_ids[: offset - self.token_start]

        new_mem_indices = None
        if self._mem_indices is not None:
            new_mem_indices = self._mem_indices[: offset - self.real_token_start]

        return ExKVCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=self._offset,
            token_ids=new_token_ids,
            mem_indices=new_mem_indices,
            storage_config=self._storage_config,
            kv_length_per_token=self._kv_length_per_token,
        )

    def update_kv_info(
        self, storage_config: HiCacheStorageConfig, kv_length_per_token: int
    ) -> ExKVCache:
        """
        Returns a new ExKVCache instance with the specified storage configuration.
        """

        if self._storage_config is not None and self._kv_length_per_token is not None:
            raise ValueError("KV info has already been set.")

        return ExKVCache(
            segments=list(self._segments),
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=self._mem_indices,
            storage_config=storage_config,
            kv_length_per_token=kv_length_per_token,
        )

    def check_token_aligned(self, aligned_size: int):
        """Checks if all segment starts and lengths are aligned to `aligned_size`.

        Raises:
            ValueError: If any segment is misaligned.
        """
        for seg in self._segments:
            if seg.token_start % aligned_size != 0:
                raise ValueError(
                    f"ExKVCache segment token_start={seg.token_start} is not aligned to {aligned_size}"
                )
            if seg.token_length % aligned_size != 0:
                raise ValueError(
                    f"ExKVCache segment token_length={seg.token_length} is not aligned to {aligned_size}"
                )

    def prefetch(
        self,
        evloop: EventLoop,
        mem_pool: KVCache | HostKVCache,
        handler: Callable[[Optional[Exception]], None],
    ):
        """
        Asynchronously loads KV cache data from storage into memory pool.

        Requires mem_indices to be bound and storage_config to be set.

        Args:
            mem_pool: The memory pool to load data into.

        Raises:
            ValueError: If mem_indices are not bound or storage_config is not set or check fails.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")

        if self._storage_config is None:
            raise ValueError("storage_config is not set")

        self.check_token_aligned(mem_pool.page_size)

        kv_cache = self._shift_real_token()

        count = len(kv_cache)
        exception = None

        def prefetch_complete(e: Optional[Exception] = None):
            nonlocal count, exception
            if exception is None:
                exception = e
            count -= 1
            if count == 0:
                handler(exception)

        # Synchronize tensor `_mem_indices`, which is being asynchronously
        # executed from the main thread.
        # The main thread and the I/O thread use different streams,
        # even though they currently share the same device.
        current_platform.synchronize()

        for seg in kv_cache:
            mem_indices = self._mem_indices[seg.token_start : seg.token_end]
            storage, filepath = ExKVCacheStorageManager.get_storage(
                seg.kv_uri, self._storage_config, mem_pool
            )
            storage.load(
                evloop,
                filepath,
                seg.kv_start,
                mem_pool,
                mem_indices,
                prefetch_complete,
            )

    def backup(self, evloop: EventLoop, mem_pool: KVCache, handler: Callable):
        """
        Saves KV cache data from memory pool to storage.

        Requires mem_indices to be bound and storage_config to be set.

        Args:
            mem_pool: The memory pool containing the data.

        Raises:
            ValueError: If mem_indices are not bound or storage_config is not set.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")

        if self._storage_config is None:
            raise ValueError("storage_config is not set")

        kv_cache = self._shift_real_token()

        count = len(kv_cache)
        exception = None

        def backup_complete(e: Optional[Exception] = None):
            nonlocal count, exception
            if exception is None:
                exception = e
            count -= 1
            if count == 0:
                handler(exception)

        # Synchronize tensor `_mem_indices`, which is being asynchronously
        # executed from the main thread.
        # The main thread and the I/O thread use different streams,
        # even though they currently share the same device.
        current_platform.synchronize()

        for seg in kv_cache:
            mem_indices = self._mem_indices[seg.token_start : seg.token_end]

            storage, filepath = ExKVCacheStorageManager.get_storage(
                seg.kv_uri, self._storage_config, mem_pool
            )

            storage.save(
                evloop,
                filepath,
                seg.kv_start,
                mem_pool,
                mem_indices,
                backup_complete,
            )


class ExKVCacheOperationKind(Enum):
    PREFETCH = auto()
    WRITEBACK = auto()


class ExKVCacheOperation:
    def __init__(
        self,
        exkvcache_id: str,
        exkvcache: ExKVCache,
        kind: ExKVCacheOperationKind,
    ):
        self.exkvcache_id = exkvcache_id
        self.exkvcache = exkvcache
        self.kind = kind
        self.done = False
        self.ok = False
        self.skipped = False

    def finish(self, ok: bool):
        self.done = True
        self.ok = ok
        self.skipped = False

    def skip(self):
        self.skipped = True


class ExKVCacheOperationStatus(Enum):
    PENDING = auto()
    INPROGRESS = auto()
    COMPLETED = auto()


class ExKVCacheController:
    class ExKVCacheConfig:
        prefetch_threshold: int = 256

    def __init__(
        self,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        config: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = self._parse_config(config)
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self._storage_config = None
        self.prefetch_size = 0

        if current_platform.is_cuda():
            current_device_index = torch.cuda.current_device()
        else:
            raise RuntimeError("Explicit KVCache only support CUDA platform")

        if device:
            self.io_device = torch.device(device, index=current_device_index)
        else:
            self.io_device = torch.device("cpu")

        self._initialize()

    def _parse_config(self, config: Optional[str] = None) -> ExKVCacheConfig:
        if config is None:
            return self.ExKVCacheConfig()
        config_dict = json.loads(config)
        return self.ExKVCacheConfig(**config_dict)

    @abstractmethod
    def prefetch_lock(self, node: TreeNode):
        raise NotImplementedError

    @abstractmethod
    def prefetch_unlock(self, node: TreeNode):
        raise NotImplementedError

    @abstractmethod
    def writeback_lock(self, node: TreeNode):
        raise NotImplementedError

    @abstractmethod
    def writeback_unlock(self, node: TreeNode):
        raise NotImplementedError

    @abstractmethod
    def mem_alloc(self, size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def mem_free(self, mem: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_size_per_token(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _insert(
        self,
        last_node: TreeNode,
        token_ids: List[int],
        mem_indices: torch.Tensor,
        extra_key: Optional[str] = None,
        priority: int = 0,
    ):
        raise NotImplementedError

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        raise NotImplementedError

    @abstractmethod
    def get_mem_pool(self) -> KVCache | HostKVCache:
        raise NotImplementedError

    @abstractmethod
    def get_mem_pool_device(self) -> KVCache:
        raise NotImplementedError

    @abstractmethod
    def generate_storage_config(self) -> HiCacheStorageConfig:
        raise NotImplementedError

    @property
    def storage_config(self) -> HiCacheStorageConfig:
        if self._storage_config is None:
            self._storage_config = self.generate_storage_config()
        return self._storage_config

    def _initialize(self):
        self.ongoing_prefetch = {}
        self.ongoing_writeback = []

        self.evloop = EventLoop("exkvcache_loop")

        self._init_io_queue()

        self.io_thread = threading.Thread(target=self._run_io_loop, daemon=True)
        self.io_thread.start()

    def _run_io_loop(self):
        current_platform.set_device(self.io_device)
        self.evloop.run()

    def _init_io_queue(self):
        self.io_queue = queue.Queue()
        self.eventfd = os.eventfd(0, flags=os.EFD_CLOEXEC | os.EFD_NONBLOCK)

        def _eventfd_handler(fd: int, event_type: EventType, data: Any = None):
            assert fd == self.eventfd, f"fd {fd} is not eventfd {self.eventfd}"
            os.eventfd_read(fd)
            self._handle_ioqueue()

        self.evloop.register_fd(self.eventfd, EventType.READ, _eventfd_handler)

    def _handle_ioqueue(self):
        while not self.io_queue.empty():
            try:
                op: ExKVCacheOperation
                op = self.io_queue.get_nowait()

                def new_op_complete(op):
                    def op_complete(exception: Optional[Exception] = None) -> None:
                        if exception is None:
                            op.finish(ok=True)
                        else:
                            op.finish(ok=False)
                            logger.error(
                                f"handle op {op.exkvcache_id} {op.kind} error: {exception}"
                            )

                    return op_complete

                if op.kind == ExKVCacheOperationKind.PREFETCH:
                    op.exkvcache.prefetch(
                        self.evloop, self.get_mem_pool(), new_op_complete(op)
                    )
                elif op.kind == ExKVCacheOperationKind.WRITEBACK:
                    op.exkvcache.backup(
                        self.evloop, self.get_mem_pool_device(), new_op_complete(op)
                    )
            except queue.Empty:
                return

    def stop(self):
        self.evloop.stop()

        if self.io_thread.is_alive():
            self.io_thread.join()

        while not self.io_queue.empty():
            try:
                op = self.io_queue.get_nowait()
                op.finish(ok=False)
            except queue.Empty:
                break

    def reset(self):
        self.stop()
        self._initialize()

    def _operation_result(self, op: ExKVCacheOperation) -> tuple[bool, bool]:
        if self.tp_world_size <= 1:
            return (op.done, op.ok)

        res = torch.tensor(
            [int(op.done), int(op.ok)], device=self.device, dtype=torch.int
        )
        torch.distributed.all_reduce(
            res,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )
        return (res[0].item() == 1, res[1].item() == 1)

    def _post(self, operation: ExKVCacheOperation):
        self.io_queue.put(operation)
        os.eventfd_write(self.eventfd, 1)

    def _skip_prefetch(self, req: Req):
        operation = ExKVCacheOperation(
            req.exkvcache_id, req.stored_exkvcache, ExKVCacheOperationKind.PREFETCH  # type: ignore
        )
        operation.skip()
        self.ongoing_prefetch[req.exkvcache_id] = (req.last_node, operation)

    def _post_prefetch(self, req: Req, stored_exkvcache: ExKVCache):
        operation = ExKVCacheOperation(
            req.exkvcache_id, stored_exkvcache, ExKVCacheOperationKind.PREFETCH  # type: ignore
        )
        self._post(operation)
        self.ongoing_prefetch[req.exkvcache_id] = (req.last_node, operation)

    def inc_prefetch_size(self, size: int):
        pass

    def dec_prefetch_size(self, size: int):
        pass

    def set_req_prefetching_len(self, req: Req, len: int):
        req.prefetching_len = 0

    def prefetch(self, req: Req, skip_if_no_mem: bool) -> bool:
        self.prefetch_lock(req.last_node)

        matched_len = len(req.prefix_indices) + req.host_hit_length
        stored_exkvcache = req.stored_exkvcache.truncate_prefix(matched_len)

        if stored_exkvcache.token_length <= self.config.prefetch_threshold:
            self._skip_prefetch(req)
            return True

        prefetch_tokens = req.full_untruncated_fill_ids[
            stored_exkvcache.token_start : stored_exkvcache.token_end
        ]
        stored_exkvcache = stored_exkvcache.bind_token_ids(prefetch_tokens)

        mem_indices = self.mem_alloc(stored_exkvcache.real_token_length)
        if mem_indices is None:
            if skip_if_no_mem:
                self._skip_prefetch(req)
                return True
            else:
                self.prefetch_unlock(req.last_node)
                return False

        self.inc_prefetch_size(stored_exkvcache.token_length)
        self.set_req_prefetching_len(req, stored_exkvcache.token_length)

        stored_exkvcache = stored_exkvcache.bind_mem_indices(mem_indices)
        stored_exkvcache = stored_exkvcache.update_kv_info(
            self.storage_config, self.get_size_per_token()
        )

        self._post_prefetch(req, stored_exkvcache)

        return True

    def check_exkvcache_prefetch_progress(
        self, exkvcache_id: str, pop: bool = True
    ) -> ExKVCacheOperationStatus:
        if exkvcache_id not in self.ongoing_prefetch:
            return ExKVCacheOperationStatus.PENDING

        last_node, operation = self.ongoing_prefetch[exkvcache_id]
        operation: ExKVCacheOperation

        if operation.skipped:
            if pop:
                self.prefetch_unlock(last_node)
                del self.ongoing_prefetch[exkvcache_id]
            return ExKVCacheOperationStatus.COMPLETED

        operation_done, operation_ok = self._operation_result(operation)
        if not operation_done:
            return ExKVCacheOperationStatus.INPROGRESS

        if not pop:
            return ExKVCacheOperationStatus.COMPLETED

        stored_exkvcache: ExKVCache = operation.exkvcache
        need_free = len(stored_exkvcache.mem_indices)

        if operation_ok:
            insert_token_ids = stored_exkvcache.token_ids
            insert_mem_indices = stored_exkvcache.mem_indices

            need_free = self._insert(
                last_node,
                insert_token_ids,
                insert_mem_indices,
            )

        self.mem_free(stored_exkvcache.real_mem_indices_prefix(need_free))
        self.dec_prefetch_size(stored_exkvcache.token_length)

        self.prefetch_unlock(last_node)
        del self.ongoing_prefetch[exkvcache_id]
        return ExKVCacheOperationStatus.COMPLETED

    def writeback(
        self,
        exkvcache_id: Optional[str],
        fresh_exkvcache: ExKVCache,
        radix_key: RadixKey,
    ) -> ExKVCache:
        if exkvcache_id is None:
            return ExKVCache()

        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        device_indices, last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        fresh_exkvcache = fresh_exkvcache.truncate_suffix(len(device_indices))

        if fresh_exkvcache.token_length == 0:
            return ExKVCache()

        self.writeback_lock(last_node)

        new_indices = device_indices[
            fresh_exkvcache.token_start : fresh_exkvcache.token_end
        ]
        fresh_exkvcache = fresh_exkvcache.bind_mem_indices(new_indices)
        fresh_exkvcache = fresh_exkvcache.update_kv_info(
            self.storage_config, self.get_size_per_token()
        )
        operation = ExKVCacheOperation(
            exkvcache_id,
            fresh_exkvcache,
            ExKVCacheOperationKind.WRITEBACK,
        )

        self._post(operation)

        self.ongoing_writeback.append((last_node, operation))

        return operation.exkvcache.result()

    def check_exkvcache_writeback_events(self) -> int:
        to_finalize = []
        for i, (node, operation) in enumerate(self.ongoing_writeback):
            operation_done, _ = self._operation_result(operation)
            if not operation_done:
                break
            to_finalize.append(i)
            self.writeback_unlock(node)

        for i in sorted(to_finalize, reverse=True):
            del self.ongoing_writeback[i]

        return len(to_finalize)


class ExRadixCache(RadixCache, ExKVCacheController):
    def __init__(self, params: CacheInitParams, config: Optional[str] = None):
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.attn_cp_rank = 0
        self.attn_cp_size = 1
        self.kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.enable_metrics = params.enable_metrics

        ExKVCacheController.__init__(
            self,
            tp_group=params.tp_cache_group,
            config=config,
            device=params.token_to_kv_pool_allocator.device,
        )
        RadixCache.__init__(self, params)

    def prefetch_lock(self, node: TreeNode):
        self.inc_lock_ref(node)

    def prefetch_unlock(self, node: TreeNode):
        self.dec_lock_ref(node)

    def writeback_lock(self, node: TreeNode):
        self.inc_lock_ref(node)

    def writeback_unlock(self, node: TreeNode):
        self.dec_lock_ref(node)

    def inc_prefetch_size(self, size: int):
        self.prefetch_size += size

    def dec_prefetch_size(self, size: int):
        self.prefetch_size -= size

    def set_req_prefetching_len(self, req: Req, len: int):
        req.prefetching_len = len

    def mem_alloc(self, size: int) -> Optional[torch.Tensor]:
        indices = self.token_to_kv_pool_allocator.alloc(size)
        if indices is None:
            self.evict(EvictParams(num_tokens=size))
            indices = self.token_to_kv_pool_allocator.alloc(size)
        return indices

    def mem_free(self, mem: torch.Tensor):
        self.token_to_kv_pool_allocator.free(mem)

    def get_size_per_token(self) -> int:
        return self.kvcache.get_size_per_token()

    def _insert(
        self,
        last_node: TreeNode,
        token_ids: List[int],
        mem_indices: torch.Tensor,
        extra_key: Optional[str] = None,
        priority: int = 0,
    ) -> int:
        res = self.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=extra_key),
                value=mem_indices,
                priority=priority,
            )
        )

        return res.prefix_len

    def generate_storage_config(self) -> HiCacheStorageConfig:
        if is_dp_attention_enabled():
            self.tp_rank = get_parallel().attn_tp_rank
            self.tp_size = get_parallel().attn_tp_size
        else:
            self.tp_rank = get_parallel().tp_rank
            self.tp_size = get_parallel().tp_size

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=self.attn_cp_rank,
            attn_cp_size=self.attn_cp_size,
            is_mla_model=isinstance(self.kvcache, MLATokenToKVPool),
            is_page_first_layout=False,
            model_name=None,
            enable_storage_metrics=self.enable_metrics,
        )

    def get_mem_pool(self) -> KVCache | HostKVCache:
        return self.kvcache

    def get_mem_pool_device(self) -> KVCache:
        return self.kvcache

    def reset(self):
        RadixCache.reset(self)
        ExKVCacheController.reset(self)

    def writeback_to_exkvcache(
        self,
        exkvcache_id: Optional[str],
        fresh_exkvcache: ExKVCache,
        radix_key: RadixKey,
    ) -> ExKVCache:
        return self.writeback(exkvcache_id, fresh_exkvcache, radix_key)

    def need_feeding(self, waiting_queue: List[Req]) -> bool:
        if self.check_exkvcache_writeback_events() > 0:
            return True

        if len(waiting_queue) > 0 and waiting_queue[0].prefetching:
            return True

        return False

    def update_exkvcache_prefetch_status(self, req: Req):
        if req.exkvcache_id is None:
            return

        if not req.prefetching:
            return

        req.prefetch_status = self.check_exkvcache_prefetch_progress(req.exkvcache_id)
        if req.prefetch_status == ExKVCacheOperationStatus.PENDING:
            raise RuntimeError(
                f"Prefetching queue should not contain pending request ({req.exkvcache_id})"
            )

        if req.prefetched:
            # Release the lock added by `add_one_req` when the req is prefetched
            self.dec_lock_ref(req.last_node)
            req.prefetching_len = 0

    def prefetch_from_exkvcache(self, req: Req):
        return self.prefetch(req, skip_if_no_mem=False)

    def revise_exkvcache_adder_result(
        self, adder: PrefillAdder, req: Req, res: AddReqResult
    ) -> AddReqResult:
        if (
            req.exkvcache_id is None
            or req not in set(adder.can_run_list)
            or req.prefetched
        ):
            return res

        # Update prefill budget without joining the running batch
        adder.new_chunked_req = None
        adder.can_run_list.remove(req)

        if req.prefetching:
            # Release the extra lock added by `add_one_req` when the req is prefetching
            self.dec_lock_ref(req.last_node)
            return res

        inprogress = self.prefetch_from_exkvcache(req)
        if not inprogress:
            from sglang.srt.managers.schedule_policy import AddReqResult

            self.dec_lock_ref(req.last_node)
            return AddReqResult.OTHER

        # Hold the lock added by `add_one_req` until the req is prefetched
        # NOT self.dec_lock_ref(req.last_node)
        req.prefetch_status = ExKVCacheOperationStatus.INPROGRESS

        return res


class ExHiRadixCache(HiRadixCache, ExKVCacheController):
    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
        config: Optional[str] = None,
    ) -> None:
        ExKVCacheController.__init__(
            self,
            tp_group=params.tp_cache_group,
            config=config,
            device=params.token_to_kv_pool_allocator.device,
        )
        HiRadixCache.__init__(self, params, server_args)

    def prefetch_lock(self, node: TreeNode):
        node.protect_host()

    def prefetch_unlock(self, node: TreeNode):
        node.release_host()

    def writeback_lock(self, node: TreeNode):
        self.inc_lock_ref(node)

    def writeback_unlock(self, node: TreeNode):
        self.dec_lock_ref(node)

    def mem_alloc(self, size: int) -> Optional[torch.Tensor]:
        host_indices = self.token_to_kv_pool_host.alloc(size)
        if host_indices is None:
            self.evict_host(size)
            host_indices = self.token_to_kv_pool_host.alloc(size)
        return host_indices

    def mem_free(self, mem: torch.Tensor):
        self.token_to_kv_pool_host.free(mem)

    def get_size_per_token(self) -> int:
        return self.token_to_kv_pool_host.get_size_per_token()

    def _insert(
        self,
        last_node: TreeNode,
        token_ids: List[int],
        mem_indices: torch.Tensor,
        extra_key: Optional[str] = None,
        priority: int = 0,
    ) -> int:
        from sglang.srt.mem_cache.utils import get_hash_str

        last_hash = last_node.get_last_hash_value()
        hash_value = get_hash_str(token_ids, last_hash, self.page_size)

        return self._insert_helper_host(
            last_node,
            RadixKey(
                token_ids=token_ids,
                extra_key=last_node.key.extra_key,
            ),
            host_value=mem_indices,
            hash_value=hash_value,
        )

    def generate_storage_config(self) -> HiCacheStorageConfig:
        return self.cache_controller._generate_storage_config()

    def get_mem_pool(self) -> KVCache | HostKVCache:
        return self.token_to_kv_pool_host

    def get_mem_pool_device(self) -> KVCache:
        return self.token_to_kv_pool_allocator.get_kvcache()

    def reset(self):
        HiRadixCache.reset(self)
        ExKVCacheController.reset(self)

    def prefetch_from_exkvcache(self, req: Req) -> bool:
        return self.prefetch(req, skip_if_no_mem=True)

    def writeback_to_exkvcache(
        self,
        exkvcache_id: Optional[str],
        fresh_exkvcache: ExKVCache,
        radix_key: RadixKey,
    ) -> ExKVCache:
        return self.writeback(exkvcache_id, fresh_exkvcache, radix_key)

    def need_feeding(self, waiting_queue: List[Req]) -> bool:
        self.check_exkvcache_writeback_events()
        return False

    def update_exkvcache_prefetch_status(self, req: Req):
        if req.exkvcache_id is None:
            return

        if req.prefetched:
            return

        req.prefetch_status = self.check_exkvcache_prefetch_progress(req.exkvcache_id)

    def revise_exkvcache_adder_result(
        self, adder: PrefillAdder, req: Req, res: AddReqResult
    ) -> AddReqResult:
        return res


class ExKVCacheStorageManager:
    """
    Manages access to storage backends, caching them per-thread to avoid recreation overhead.

    Uses thread-local storage to hold backend instances keyed by backend type and configuration.
    """

    _local = threading.local()
    _lock = threading.Lock()

    @classmethod
    def _get_thread_cache(
        cls,
    ) -> Dict[Tuple[str, frozenset], Tuple[HiCacheStorage, bool, bool]]:
        """Get or create the thread-local cache dict."""
        if not hasattr(cls._local, "cache"):
            cls._local.cache = {}
        return cls._local.cache

    @classmethod
    def get_storage(
        cls,
        uri: str,
        storage_config: HiCacheStorageConfig,
        mem_pool: KVCache | HostKVCache,
        **kwargs,
    ) -> Tuple[HiCacheStorage, str]:
        """
        Retrieves a storage backend instance for a given URI.

        Caches the instance per-thread based on backend type and parsed configuration.

        Args:
            uri: The URI specifying the backend and resource path.
            storage_config: Base configuration for storage.
            mem_pool: Memory pool for the backend.
            **kwargs: Additional arguments for backend creation.

        Returns:
            A tuple of (HiCacheStorage instance, extracted filepath from URI).

        Raises:
            ValueError: If the URI is invalid or the backend cannot be found/created.
        """
        cache = cls._get_thread_cache()

        backend_name = parse_connector_type(uri)
        if not backend_name:
            raise ValueError(f"Invalid URI: missing backend name in '{uri}'")

        backend_class = StorageBackendFactory.get_backend_class(backend_name)

        extra_config, filepath = backend_class.parse_uri(uri)
        storage_config = replace(storage_config, extra_config=extra_config)
        cache_key = (
            backend_name,
            frozenset(extra_config.items()) if extra_config else frozenset(),
        )

        with cls._lock:
            backend, device_registered, host_registered = cache.get(
                cache_key, (None, False, False)
            )
            if backend is None:
                backend = StorageBackendFactory.create_backend(
                    backend_name=backend_name,
                    storage_config=storage_config,
                    mem_pool_host=None,
                    **kwargs,
                )

            if not device_registered and isinstance(mem_pool, KVCache):
                backend.register_mem_pool_device(mem_pool)
                device_registered = True
                cache[cache_key] = (backend, device_registered, host_registered)

            if not host_registered and isinstance(mem_pool, HostKVCache):
                backend.register_mem_pool_host(mem_pool)
                host_registered = True
                cache[cache_key] = (backend, device_registered, host_registered)

        return backend, filepath
