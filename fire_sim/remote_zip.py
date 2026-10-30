"""Read individual members of a large remote zip via HTTP range requests (curl handles TLS)."""

from __future__ import annotations

import argparse
import io
import re
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        headers = subprocess.run(
            ["curl", "-sIL", url], check=True, capture_output=True, text=True
        ).stdout
        self.size = int(re.findall(r"content-length:\s*(\d+)", headers, re.IGNORECASE)[-1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.position + size, self.size) - 1
        data = _curl_range(self.url, self.position, end)
        self.position += len(data)
        return data


def matching_members(url: str, patterns: list[str]) -> list[zipfile.ZipInfo]:
    """Zip entries (files only) whose name contains any of the patterns."""
    archive = zipfile.ZipFile(HttpRangeFile(url))
    return [
        info
        for info in archive.infolist()
        if not info.is_dir() and (not patterns or any(p in info.filename for p in patterns))
    ]


def extract_members(
    url: str, patterns: list[str], dest_dir: Path, *, strip_prefix: str = ""
) -> list[Path]:
    """Download only the matching members, each with a single range request."""
    written = []
    for info in matching_members(url, patterns):
        relative = info.filename.removeprefix(strip_prefix)
        target = dest_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_fetch_member(url, info))
        written.append(target)
    return written


def _fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    local_header = _curl_range(url, info.header_offset, info.header_offset + 29)
    name_length, extra_length = struct.unpack("<HH", local_header[26:30])
    data_start = info.header_offset + 30 + name_length + extra_length
    compressed = _curl_range(url, data_start, data_start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED:
        return compressed
    return zlib.decompress(compressed, -zlib.MAX_WBITS)


def _curl_range(url: str, start: int, end: int) -> bytes:
    return subprocess.run(
        ["curl", "-sL", "-r", f"{start}-{end}", url], check=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("patterns", nargs="*")
    args = parser.parse_args()
    for info in matching_members(args.url, args.patterns):
        print(f"{info.file_size:>12}  {info.filename}")


if __name__ == "__main__":
    main()
