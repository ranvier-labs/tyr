"""Regenerate the small dtype promotion fixture using only Python's stdlib."""

import json
from pathlib import Path
import struct


entries = [
    ("Bool", "BOOL", struct.pack("?", True)),
    ("UInt8", "U8", struct.pack("B", 255)),
    ("Int8", "I8", struct.pack("b", -1)),
    ("Int16", "I16", struct.pack("<h", 1)),
    ("Int32", "I32", struct.pack("<i", 1)),
    ("Int64", "I64", struct.pack("<q", 1)),
    ("Float16", "F16", struct.pack("<e", 1.0)),
    ("BFloat16", "BF16", struct.pack("<H", 0x3F80)),
    ("Float32", "F32", struct.pack("<f", 1.0)),
    ("Float64", "F64", struct.pack("<d", 1.0)),
    ("Float8E4M3FN", "F8_E4M3", bytes([0x38])),
    ("Float8E5M2", "F8_E5M2", bytes([0x3C])),
]
header = {}
payload = bytearray()
for name, dtype, data in entries:
    offset = len(payload)
    payload.extend(data)
    header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, len(payload)]}
encoded = json.dumps(header, separators=(",", ":")).encode()
encoded += b" " * (-len(encoded) % 8)
Path(__file__).with_name("dtype_scalars.safetensors").write_bytes(
    struct.pack("<Q", len(encoded)) + encoded + payload
)
