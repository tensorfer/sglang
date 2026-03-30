from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExKVCacheSegment:
    token_start: int
    token_length: int
    kv_uri: str
    kv_start: int
    kv_length: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length

    @property
    def kv_end(self) -> Optional[int]:
        return self.kv_start + self.kv_length if self.kv_length is not None else None


class ExKVCache:
    def __init__(
        self,
        segments: Optional[List[Dict]] = None,
    ):
        if segments is None or not segments:
            self._segments: List[ExKVCacheSegment] = []
            return

        _segments = [ExKVCacheSegment(**seg) for seg in segments]

        sorted_segments = sorted(_segments, key=lambda s: s.token_start)

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

        self._segments = sorted_segments

    def to_dicts(self) -> List[Dict]:
        return [seg.to_dict() for seg in self._segments]

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def token_end(self) -> Optional[int]:
        return self._segments[-1].token_end if self._segments else None

    @property
    def kv_end(self) -> Optional[int]:
        return self._segments[-1].kv_end if self._segments else None

    def extend(self, other: "ExKVCache", strict: bool = True) -> None:
        if not other._segments:
            return
        if not self._segments:
            self._segments.extend(other._segments)
            return
        if strict:
            expected = self.token_end
            actual = other._segments[0].token_start
            if expected != actual:
                raise ValueError(
                    f"Token discontinuity: current ends at {expected}, "
                    f"other starts at {actual}."
                )
        self._segments.extend(other._segments)

    def __iadd__(self, other: "ExKVCache") -> "ExKVCache":
        if not isinstance(other, ExKVCache):
            return NotImplemented
        self.extend(other, strict=True)
        return self


class ExKVCacheBackend:
    def __init__(self, kvcache_id: str, dir: str):
        self.kvcache_id = kvcache_id
        self.dir = dir
        self.stored_kvcache = ExKVCache()
        self.fresh_kvcache = ExKVCache()

    def prepare_exkvcache_file(self, filename: str) -> str:
        """Create and return explicit kvcache file path"""
        raise NotImplementedError()

    def file_scheme(self, filename: str) -> str:
        """Return file scheme for kvcache uri"""
        raise NotImplementedError()

    def kvcache_params(self, new_token_length: int, filename: str) -> Dict:
        fresh_kvcache = []

        if len(self.stored_kvcache) == 0:
            fresh_kvcache = [
                {
                    "token_start": 0,
                    "token_length": new_token_length,
                    "kv_uri": self.file_scheme(filename),
                    "kv_start": 0,
                }
            ]
        else:
            fresh_kvcache = [
                {
                    "token_start": self.stored_kvcache.token_end,
                    "token_length": new_token_length,
                    "kv_uri": self.file_scheme(filename),
                    "kv_start": self.stored_kvcache.kv_end,
                }
            ]

        return {
            "id": self.kvcache_id,
            "stored_kvcache": self.stored_kvcache.to_dicts(),
            "fresh_kvcache": fresh_kvcache,
        }

    def update_kvcache(self, fresh_kvcache: List[Dict]):
        self.stored_kvcache += ExKVCache(fresh_kvcache)


class ExKVCacheFileBackend(ExKVCacheBackend):
    def prepare_exkvcache_file(self, filename: str) -> str:
        path = Path(self.dir) / filename
        path.touch()
        return str(path)

    def file_scheme(self, filename: str) -> str:
        return f"file:///{filename}"
