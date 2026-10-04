"""Low-memory streaming reader for large JSON files (Track B canonicalization).

iter_json_array(path)         -> yields each top-level element of a JSON array file
iter_json_array_at(path, key) -> yields each element of the array stored under top-level `key`
                                 (e.g. SQuAD {"version":..., "data":[...]})
iter_jsonl(path)              -> yields each non-empty line parsed as JSON

Pure stdlib (json.JSONDecoder.raw_decode over a sliding buffer). Peak RSS ~ one element + buffer,
so 2wiki train.json (708 MB) streams in well under 300 MB instead of ~4 GB for json.load.
"""
import json

_DEC = json.JSONDecoder()
_WS = " \t\r\n"


def _skip_ws(buf, pos):
    n = len(buf)
    while pos < n and buf[pos] in _WS:
        pos += 1
    return pos


def _iter_array_from(fh, buf, pos, chunk=1 << 20):
    """buf[pos] must be '['. Yields decoded elements; returns (buf, pos) after the closing ']'."""
    assert buf[pos] == "[", f"expected '[' at {pos}, got {buf[pos:pos+20]!r}"
    pos += 1
    while True:
        # make sure we have some data to look at
        if pos >= len(buf) - 1:
            more = fh.read(chunk)
            if not more and pos >= len(buf):
                raise ValueError("unterminated JSON array")
            buf = buf[pos:] + more
            pos = 0
        pos = _skip_ws(buf, pos)
        if pos >= len(buf):
            more = fh.read(chunk)
            if not more:
                raise ValueError("unterminated JSON array")
            buf = buf[pos:] + more
            pos = 0
            continue
        c = buf[pos]
        if c == "]":
            return buf, pos + 1
        if c == ",":
            pos += 1
            continue
        while True:
            try:
                obj, end = _DEC.raw_decode(buf, pos)
                break
            except json.JSONDecodeError:
                more = fh.read(chunk)
                if not more:
                    raise
                buf = buf[pos:] + more
                pos = 0
        yield obj
        pos = end


def iter_json_array(path, encoding="utf-8", chunk=1 << 20):
    with open(path, "r", encoding=encoding) as fh:
        buf = fh.read(chunk)
        pos = _skip_ws(buf, 0)
        yield from _iter_array_from(fh, buf, pos, chunk)


def iter_json_array_at(path, key, encoding="utf-8", chunk=1 << 20):
    """Stream the array stored at top-level object key `key`; other keys are skipped (they must
    precede the array or be small). Elements before `key` are decoded and discarded."""
    with open(path, "r", encoding=encoding) as fh:
        buf = fh.read(chunk)
        pos = _skip_ws(buf, 0)
        assert buf[pos] == "{", "expected top-level object"
        pos += 1
        while True:
            pos = _skip_ws(buf, pos)
            if pos >= len(buf):
                more = fh.read(chunk)
                if not more:
                    raise ValueError(f"key {key!r} not found")
                buf = buf[pos:] + more; pos = 0
                continue
            c = buf[pos]
            if c == ",":
                pos += 1; continue
            if c == "}":
                raise ValueError(f"key {key!r} not found")
            # key string
            while True:
                try:
                    k, end = _DEC.raw_decode(buf, pos); break
                except json.JSONDecodeError:
                    more = fh.read(chunk)
                    if not more:
                        raise
                    buf = buf[pos:] + more; pos = 0
            pos = _skip_ws(buf, end)
            assert buf[pos] == ":", "malformed object"
            pos = _skip_ws(buf, pos + 1)
            if pos >= len(buf):
                buf = buf[pos:] + fh.read(chunk); pos = 0
            if k == key:
                assert buf[pos] == "[", f"value at {key!r} is not an array"
                yield from _iter_array_from(fh, buf, pos, chunk)
                return
            # skip a non-target value
            while True:
                try:
                    _, end = _DEC.raw_decode(buf, pos); break
                except json.JSONDecodeError:
                    more = fh.read(chunk)
                    if not more:
                        raise
                    buf = buf[pos:] + more; pos = 0
            pos = end


def iter_jsonl(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
