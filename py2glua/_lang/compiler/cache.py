from __future__ import annotations

import hashlib
import pickle
import shutil
import sys
from pathlib import Path
from typing import BinaryIO

from ...config import Py2GluaConfig
from ..py.ir_builder import PyIRFile


def _read_u32(f: BinaryIO) -> int:
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("Unexpected EOF while reading u32")

    return int.from_bytes(data, "little")


def _write_u32(f: BinaryIO, value: int) -> None:
    f.write(value.to_bytes(4, "little"))


def _hash_file(path: Path) -> bytes:
    h = hashlib.blake2s(digest_size=16)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)

    return h.digest()


def _hash_path(path: Path) -> str:
    return hashlib.blake2s(
        str(path).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


class _IndexCache:
    """
    Binary format:

    HEADER:
      16 bytes  global_fingerprint
      u32       entry_count

    BODY (repeat entry_count):
      16 bytes  source_hash
      u32       path_len
      bytes     utf-8 encoded absolute path
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[Path, bytes] = {}
        self._dirty: dict[Path, bytes] = {}

    def _build_global_fingerprint(self, fp_name: list[str]) -> bytes:
        parts = [
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}",
            Py2GluaConfig.version(),
            *fp_name,
        ]
        return hashlib.blake2s(
            "\n".join(parts).encode("utf-8"),
            digest_size=16,
        ).digest()

    def _gc_full(self) -> None:
        try:
            cache_dir = self.path.parent
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        except Exception:
            pass

    def load(self, fp_name: list[str]) -> None:
        self._entries.clear()

        if not self.path.exists():
            return

        expected_fp = self._build_global_fingerprint(fp_name)

        try:
            with self.path.open("rb") as f:
                if f.read(16) != expected_fp:
                    self._gc_full()
                    return

                count = _read_u32(f)

                for _ in range(count):
                    src_hash = f.read(16)
                    path_len = _read_u32(f)
                    raw = f.read(path_len)
                    if len(raw) != path_len:
                        break

                    path = Path(raw.decode("utf-8"))

                    if not path.exists() or not path.is_file():
                        continue

                    if _hash_file(path) == src_hash:
                        self._entries[path] = src_hash

        except Exception:
            self._entries.clear()
            self._gc_full()

    def mark_dirty(self, path: Path) -> None:
        self._dirty[path] = _hash_file(path)

    def save(self, fp_name: list[str]) -> None:
        if not self._dirty:
            return

        self._entries.update(self._dirty)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        global_fp = self._build_global_fingerprint(fp_name)

        with self.path.open("wb") as f:
            f.write(global_fp)
            _write_u32(f, len(self._entries))

            for path, src_hash in self._entries.items():
                raw = str(path).encode("utf-8")
                f.write(src_hash)
                _write_u32(f, len(raw))
                f.write(raw)

        self._dirty.clear()

    def valid_paths(self) -> set[Path]:
        return set(self._entries)


class IRCache:
    def __init__(self) -> None:
        self.valid_paths: set[Path] = set()
        self._index: _IndexCache | None = None
        self._fp_name: list[str] = []
        self._cache_dir: Path | None = None

    def validate(
        self,
        project_root: Path,
        file_pass: list[str],
    ) -> None:
        self._cache_dir = project_root.parent / ".plg-sdk/cache/ir_build_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._fp_name = list(file_pass)
        self._index = _IndexCache(self._cache_dir / "index.bin")
        self._index.load(self._fp_name)

        self.valid_paths = self._index.valid_paths()

        valid = {_hash_path(p) for p in self.valid_paths}

        for f in self._cache_dir.glob("*.pkl"):
            name = f.stem.replace(".raw", "")
            if name not in valid:
                f.unlink(missing_ok=True)

    def load_raw(self, path: Path) -> PyIRFile | None:
        assert self._cache_dir is not None

        raw = self._cache_dir / f"{_hash_path(path)}.raw.pkl"
        if not raw.exists():
            return None

        try:
            with raw.open("rb") as f:
                return pickle.load(f)

        except Exception:
            raw.unlink(missing_ok=True)
            return None

    def store_raw(self, path: Path, ir: PyIRFile) -> None:
        assert self._cache_dir is not None

        raw = self._cache_dir / f"{_hash_path(path)}.raw.pkl"
        if raw.exists():
            return

        with raw.open("wb") as f:
            pickle.dump(ir, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> PyIRFile | None:
        if path not in self.valid_paths:
            return None

        assert self._cache_dir is not None

        pkl = self._cache_dir / f"{_hash_path(path)}.pkl"
        if not pkl.exists():
            self.valid_paths.discard(path)
            return None

        try:
            with pkl.open("rb") as f:
                return pickle.load(f)

        except Exception:
            pkl.unlink(missing_ok=True)
            self.valid_paths.discard(path)
            return None

    def store(self, path: Path, ir: PyIRFile) -> None:
        assert self._cache_dir is not None
        assert self._index is not None

        pkl = self._cache_dir / f"{_hash_path(path)}.pkl"
        if not pkl.exists():
            with pkl.open("wb") as f:
                pickle.dump(ir, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._index.mark_dirty(path)
        self.valid_paths.add(path)

    def commit(self) -> None:
        assert self._index is not None
        self._index.save(self._fp_name)
