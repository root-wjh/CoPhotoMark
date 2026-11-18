"""Image-embedded LSB metadata channel.

Paper alignment: Algorithm 1 "optionally embed (M, sigma_meta) into Y" and
Algorithm 2 "ExtractLSB(Y)". The enrollment metadata M (including its HMAC
sigma_meta) is therefore carried INSIDE the delivered PNG, in the least
significant bits of the luminance (Y) plane. Verification still requires the
secret key and trusted expected claim. A sidecar .cpm.json is an optional debug
copy and is never opened on the default verification path.

Bit frame, written into Y-plane LSBs in raster order:
    magic b"CPM1" (32 bits) | uint32 BE payload byte length (32 bits)
    | canonical compact JSON payload (the full metadata object incl. HMAC)

Security: extraction yields raw JSON only; engine.validate_metadata re-checks
the HMAC under the domain-separated metadata subkey. Any single-bit corruption
(JPEG recompression, blur, cropping, geometry edits, pixel moves) therefore
invalidates the whole channel; verification then falls back to the blind
pipeline (metadata loss -> blind verify). Not every image edit necessarily
damages the frame, and LSB erasure can also perturb DCT evidence.

Implementation note: not every Y LSB survives OpenCV's 8-bit BGR<->YCrCb
fixed-point round trip (~0.2% flip on random pixels). For each requested bit
we search nearby luminance values (+/-1, +/-2, ... by increasing distance)
for one whose round-tripped LSB equals the requested bit. On natural images
the final bitstream is verified; there is no universal perceptual bound.
"""
from __future__ import annotations
import json
import cv2
import numpy as np
from .io import canonical, no_duplicates

MAGIC = b"CPM1"
HEADER_BITS = len(MAGIC) * 8 + 32
MAX_PAYLOAD_BYTES = 1 << 20  # engine.validate_metadata applies the same 1 MiB cap
_MAX_DELTA_ATTEMPTS = 12


def _y(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def _write_y_bits(image, bits):
    """Write bits into the first len(bits) raster LSB positions of Y.

    Guarantees every bit reads back after BGR->YCrCb reconversion; raises
    RuntimeError if a pixel cannot be placed (not observed on natural images).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1 or bits.size == 0 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be a non-empty 0/1 array")
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y0 = ycc[:, :, 0].astype(np.int16)
    flat = y0.reshape(-1)
    count = bits.size
    if count > flat.size:
        raise ValueError("Image has too few luminance pixels for the metadata frame")
    # Search nearest alternative Y values by |delta|: +1,-1,+2,-2,...
    # Deltas are not restricted to odd numbers because OpenCV's fixed-point
    # round trip has per-pixel systematic rounding bias; we accept whatever
    # nearby value actually reads back with the requested LSB.
    deltas = np.array([d for k in range(1, _MAX_DELTA_ATTEMPTS + 1)
                       for d in (k, -k)], dtype=np.int16)
    chosen = np.zeros(count, dtype=np.int16)
    attempt = np.zeros(count, dtype=np.int16)
    remaining = (flat[:count] & 1) != bits
    out = image.copy()
    for _ in range(len(deltas)):
        if not remaining.any():
            break
        chosen[remaining] = deltas[np.minimum(attempt[remaining], len(deltas) - 1)]
        attempt[remaining] += 1
        ywork = y0.copy()
        ywork.reshape(-1)[:count] = np.clip(flat[:count] + chosen, 0, 255)
        ycc[:, :, 0] = ywork.astype(np.uint8)
        converted = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        # Keep every pixel outside the frame byte-for-byte unchanged. Even
        # a nominal BGR/YCrCb round trip can modify untouched background.
        out.reshape(-1, 3)[:count] = converted.reshape(-1, 3)[:count]
        readback = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[:, :, 0].reshape(-1)[:count]
        remaining = (readback & 1) != bits
    if remaining.any():
        raise RuntimeError(f"{int(remaining.sum())} metadata bits could not be placed in Y LSBs")
    return out


def embed_metadata_lsb(image, metadata):
    """Return a copy of the BGR image with metadata embedded inside its Y LSBs."""
    payload = canonical(metadata)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Metadata exceeds the embedded-channel size cap")
    frame = MAGIC + len(payload).to_bytes(4, byteorder="big") + payload
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    return _write_y_bits(image, bits)


def _read_bits(image, count):
    flat = _y(image).reshape(-1)
    if count > flat.size:
        return None
    return (flat[:count] & 1).astype(np.uint8)


def extract_metadata_lsb(image):
    """Read and JSON-parse embedded metadata; return None when absent/corrupt.

    Never raises: a damaged channel must behave as "no metadata" so callers
    safely fall back to blind verification.
    """
    try:
        if image.ndim != 3 or image.shape[2] != 3:
            return None
        flat = _y(image).reshape(-1)
        if flat.size < HEADER_BITS:
            return None
        header = _read_bits(image, HEADER_BITS)
        magic = np.packbits(header[:len(MAGIC) * 8]).tobytes()
        if magic != MAGIC:
            return None
        length = int.from_bytes(np.packbits(header[len(MAGIC) * 8:]).tobytes(), byteorder="big")
        if length <= 0 or length > MAX_PAYLOAD_BYTES:
            return None
        total = HEADER_BITS + length * 8
        if total > flat.size:
            return None
        body_bits = _read_bits(image, total)[HEADER_BITS:]
        payload = np.packbits(body_bits).tobytes()
        metadata = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return metadata if isinstance(metadata, dict) else None
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def strip_metadata_lsb(image):
    """Erase the embedded channel (simulates a platform that sanitises it).

    Zeros the header and, when readable, the payload region. This changes
    pixels and can perturb DCT coefficients: blind recovery is measured,
    never assumed to be unaffected.
    """
    flat = _y(image).reshape(-1)
    if flat.size < HEADER_BITS:
        return image.copy()
    header = _read_bits(image, HEADER_BITS)
    length = 0
    if np.packbits(header[:len(MAGIC) * 8]).tobytes() == MAGIC:
        length = int.from_bytes(np.packbits(header[len(MAGIC) * 8:]).tobytes(), byteorder="big")
        if not 0 < length <= MAX_PAYLOAD_BYTES or HEADER_BITS + length * 8 > flat.size:
            length = 0
    count = min(flat.size, HEADER_BITS + length * 8)
    return _write_y_bits(image, np.zeros(count, dtype=np.uint8))
