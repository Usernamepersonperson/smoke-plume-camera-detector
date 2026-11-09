"""Read a remote parquet file row group by row group via HTTP range requests.

Used only by the data-preparation script, so the 1 GB training shards never
have to be stored on disk. Nothing here is needed to run the detector.
"""

from __future__ import annotations

import io
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq


class RangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:  # follows the CDN redirect
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_parquet(url: str) -> pq.ParquetFile:
    return pq.ParquetFile(pa.PythonFile(RangeFile(url), mode="r"))
