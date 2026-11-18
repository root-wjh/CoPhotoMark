from __future__ import annotations
import hashlib
import hmac
import re
import secrets
from dataclasses import asdict, dataclass
import numpy as np
from .io import canonical

VARIANTS = {"full", "no_count", "no_id", "roi_qim"}

@dataclass(frozen=True)
class Config:
    version: int = 2
    payload_bits: int = 64
    block: int = 12
    stride: int = 12
    step: int = 18
    min_roi: int = 64
    min_blocks: int = 3
    variant: str = "full"
    blind_blocks: tuple = (12, 16, 20, 24)
    max_candidates: int = 32
    blind_offsets: tuple = (0, -1, 1, -2, 2)
    min_std: int = 2
    min_pixel: int = 5
    max_pixel: int = 250
    max_faces: int = 255

    def __post_init__(self):
        fields = ("version", "payload_bits", "block", "stride", "step", "min_roi", "min_blocks",
                  "max_candidates", "min_std", "min_pixel", "max_pixel", "max_faces")
        if any(type(getattr(self, f)) is not int for f in fields):
            raise ValueError("Protocol numbers must be integers; booleans/floats forbidden")
        if self.version != 2 or self.payload_bits not in (64, 128) or self.variant not in VARIANTS:
            raise ValueError("Unsupported version, payload or variant")
        if not 12 <= self.block <= 32 or self.block % 2 or self.stride != self.block:
            raise ValueError("Enrollment requires even blocks and non-overlapping stride=block")
        if not 4 <= self.step <= 128 or not 32 <= self.min_roi <= 1024 or not 1 <= self.min_blocks <= 100:
            raise ValueError("Invalid step or qualification parameters")
        if not 1 <= self.max_candidates <= 1024 or not 1 <= self.max_faces <= 255:
            raise ValueError("Invalid search/face budget")
        if not 0 <= self.min_pixel < self.max_pixel <= 255 or not 0 <= self.min_std <= 100:
            raise ValueError("Invalid block quality rule")
        blocks = tuple(self.blind_blocks)
        if not blocks or len(blocks) > 8 or any(type(b) is not int or b < 12 or b > 32 or b % 2 for b in blocks):
            raise ValueError("Invalid blind block allowlist")
        object.__setattr__(self, "blind_blocks", blocks)
        offsets = tuple(self.blind_offsets)
        if not offsets or offsets[0] != 0 or len(offsets) > 32 or len(set(offsets)) != len(offsets) or any(type(x) is not int or abs(x)>16 for x in offsets):
            raise ValueError("Invalid bounded blind geometry offsets")
        object.__setattr__(self, "blind_offsets", offsets)
        if self.block not in blocks or len(midband(self.block)) < self.payload_bits:
            raise ValueError("Enrollment block missing from search or insufficient capacity")

    @property
    def tag_bits(self):
        return self.payload_bits - 16

    def to_dict(self):
        result = asdict(self)
        result["blind_blocks"] = list(self.blind_blocks)
        result["blind_offsets"] = list(self.blind_offsets)
        return result

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical(self.to_dict())).hexdigest()

def midband(block):
    return [(y, s-y) for s in range(max(2, block // 4), block + block // 2 + 1)
            for y in range(block) if 0 <= s-y < block and (y, s-y) != (0, 0)]

def subkey(master, domain):
    if not isinstance(master, bytes) or len(master) < 32:
        raise ValueError("Master key must contain at least 32 random bytes")
    prk = hmac.new(b"CoPhotoMark-v1-HKDF", master, hashlib.sha256).digest()
    return hmac.new(prk, b"CoPhotoMark/v1/" + domain.encode("ascii") + b"\x01", hashlib.sha256).digest()

def new_claim(message, config, image_id=None):
    claim = {"version": 1, "image_id": image_id or secrets.token_hex(16), "nonce": secrets.token_hex(16),
             "message": message, "config_sha256": config.fingerprint}
    validate_claim(claim, config)
    return claim

def validate_claim(claim, config):
    if not isinstance(claim, dict) or set(claim) != {"version", "image_id", "nonce", "message", "config_sha256"}:
        raise ValueError("Invalid expected-claim schema")
    if type(claim["version"]) is not int or claim["version"] != 1:
        raise ValueError("Unsupported claim version")
    if not isinstance(claim["image_id"], str) or not 1 <= len(claim["image_id"]) <= 128:
        raise ValueError("Invalid image identifier")
    if not isinstance(claim["nonce"], str) or not re.fullmatch(r"[0-9a-f]{32}", claim["nonce"]):
        raise ValueError("Nonce must be 128 bits, lowercase hexadecimal")
    if not isinstance(claim["message"], str) or len(claim["message"].encode("utf-8")) > 4096:
        raise ValueError("Invalid claimed message")
    if claim["config_sha256"] != config.fingerprint:
        raise ValueError("Claim belongs to another configuration")

def position_list(config, block, master, claim):
    pool = midband(block)
    if len(pool) < config.payload_bits:
        raise ValueError("Insufficient coefficient capacity")
    seed = canonical({"claim": claim, "block": block, "purpose": "positions"})
    key = subkey(master, "positions")
    pool.sort(key=lambda p: hmac.new(key, seed + bytes(p), hashlib.sha256).digest())
    return pool[:config.payload_bits]

def structural_fields(count, index, variant):
    if variant == "no_count":
        return 0, index
    if variant == "no_id":
        return count, 0
    if variant == "roi_qim":
        return 0, 0
    return count, index

def tag_bytes(count, index, config, master, claim):
    count, index = structural_fields(count, index, config.variant)
    body = {"claim": claim, "variant": config.variant, "count": count, "index": index}
    return hmac.new(subkey(master, "watermark"), canonical(body), hashlib.sha256).digest()[:config.tag_bits // 8]

def payload(count, index, config, master, claim):
    if not 1 <= index <= count <= config.max_faces:
        raise ValueError("Face count/index overflow")
    c, i = structural_fields(count, index, config.variant)
    return np.unpackbits(np.frombuffer(bytes((c, i)) + tag_bytes(count, index, config, master, claim), dtype=np.uint8))

def authenticate_metadata(body, master):
    return hmac.new(subkey(master, "metadata"), canonical(body), hashlib.sha256).hexdigest()
