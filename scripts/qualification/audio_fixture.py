"""Deterministic PCM16 input shared by the Lean and Python model checks."""
import math
import struct

TRANSFORM = "ieee_float32_to_pcm16_rne_v1"


def pcm16_wav(source, spec):
    """Preserve rate/channels; round float32 * 32768 to even, then saturate."""
    if spec["transformation"] != TRANSFORM:
        raise ValueError("Unknown audio fixture transformation")
    if len(source) < 12 or source[:4] != b"RIFF" or source[8:12] != b"WAVE":
        raise ValueError("Expected a little-endian RIFF WAV")
    if struct.unpack_from("<I", source, 4)[0] + 8 != len(source):
        raise ValueError("Invalid RIFF container length")
    chunks = {}
    position = 12
    while position < len(source):
        if position + 8 > len(source):
            raise ValueError("Truncated WAV chunk header")
        tag, size = struct.unpack_from("<4sI", source, position)
        position += 8
        end = position + size
        if end + size % 2 > len(source):
            raise ValueError("Truncated WAV chunk")
        if tag in (b"fmt ", b"data"):
            if tag in chunks:
                raise ValueError("Duplicate WAV format/data chunk")
            chunks[tag] = source[position:end]
        position = end + size % 2
    fmt, data = chunks.get(b"fmt ", b""), chunks.get(b"data", b"")
    if len(fmt) < 16:
        raise ValueError("Missing WAV format")
    kind, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", fmt)
    if kind != 3 or bits != 32:
        raise ValueError("Expected IEEE float32 WAV samples")
    if channels < 1 or channels != spec["channels"] or rate != spec["sample_rate"]:
        raise ValueError("Unexpected WAV sample rate/channels")
    if alignment != channels * 4 or byte_rate != rate * alignment:
        raise ValueError("Invalid float32 WAV frame layout")
    if not data or len(data) != spec["frames"] * alignment:
        raise ValueError("Unexpected WAV frame count")
    if spec["bits_per_sample"] != 16:
        raise ValueError("Expected a PCM16 output specification")
    samples = bytearray()
    for (value,) in struct.iter_unpack("<f", data):
        if not math.isfinite(value):
            raise ValueError("Non-finite WAV sample")
        quantized = max(-32768, min(32767, round(value * 32768)))
        samples.extend(struct.pack("<h", quantized))
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(samples),
        b"WAVE", b"fmt ", 16, 1, channels, rate, rate * channels * 2,
        channels * 2, 16, b"data", len(samples))
    return header + samples
