from __future__ import annotations
import cv2
import numpy as np
from .protocol import position_list, tag_bytes

def qim_reconstruct(values, bits, step):
    """Paper Eq. (3)-(5): Q_a={2*Delta*k+a*Delta}; dither disabled.

    step is Delta (distance between codebooks), not the lattice period.
    Protocol version 2 intentionally invalidates the earlier +/-step/4 code.
    """
    offset = np.asarray(bits) * step
    return np.rint((values - offset) / (2 * step)) * (2 * step) + offset

def luminance(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def valid_blocks(y, block, stride, config):
    for top in range(0, y.shape[0] - block + 1, stride):
        for left in range(0, y.shape[1] - block + 1, stride):
            patch = y[top:top+block, left:left+block]
            if patch.std() >= config.min_std and patch.min() >= config.min_pixel and patch.max() <= config.max_pixel:
                yield top, left

def block_count(roi, config):
    return sum(1 for _ in valid_blocks(luminance(roi)[:, :, 0], config.block, config.stride, config))

def embed_roi(roi, bits, config, master, claim):
    positions = position_list(config, config.block, master, claim)
    channel = luminance(roi)
    y = channel[:, :, 0].astype(np.float32)
    locations = list(valid_blocks(y, config.block, config.stride, config))
    iy, ix = np.array(positions).T
    for top, left in locations:
        patch = y[top:top+config.block, left:left+config.block]
        coeff = cv2.dct(patch)
        coeff[iy, ix] = qim_reconstruct(coeff[iy, ix], bits, config.step)
        y[top:top+config.block, left:left+config.block] = cv2.idct(coeff)
    channel[:, :, 0] = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return cv2.cvtColor(channel, cv2.COLOR_YCrCb2BGR), len(locations)

def decode_roi(roi, block, stride, config, master, claim, use_voting=True):
    positions = position_list(config, block, master, claim)
    y = luminance(roi)[:, :, 0].astype(np.float32)
    iy, ix = np.array(positions).T
    observations = []
    for top, left in valid_blocks(y, block, stride, config):
        values = cv2.dct(y[top:top+block, left:left+block])[iy, ix]
        q1 = qim_reconstruct(values, 1, config.step)
        q0 = qim_reconstruct(values, 0, config.step)
        observations.append((abs(values - q1) < abs(values - q0)).astype(np.uint8))
    if len(observations) < config.min_blocks:
        return {"valid": False, "reason": "insufficient_blocks", "score": 0.0, "blocks": len(observations)}
    votes = np.stack(observations)
    if use_voting:
        ones = votes.sum(axis=0)
        erased = ones * 2 == len(votes)
        recovered = (ones * 2 > len(votes)).astype(np.uint8)
    else:
        recovered, erased = votes[0], np.zeros(config.payload_bits, dtype=bool)
    count, index = np.packbits(recovered[:16]).tolist()
    valid = not bool(erased[:16].any())
    if config.variant == "full":
        valid &= 1 <= index <= count <= config.max_faces
    elif config.variant == "no_count":
        valid &= count == 0 and 1 <= index <= config.max_faces
    elif config.variant == "no_id":
        valid &= 1 <= count <= config.max_faces and index == 0
    else:
        valid &= count == index == 0
    expected_tag = np.unpackbits(np.frombuffer(tag_bytes(count, index, config, master, claim), dtype=np.uint8))
    tag_matches = (recovered[16:] == expected_tag) & ~erased[16:]
    return {"valid": bool(valid), "reason": "decoded" if valid else "invalid_or_erased_header",
            "count": count, "index": index, "score": float(tag_matches.mean()) if valid else 0.0,
            "tag_matches": int(tag_matches.sum()), "tag_bits": config.tag_bits, "blocks": len(votes),
            "erasures": int(erased.sum()), "raw_hex": np.packbits(recovered).tobytes().hex(),
            "block": block, "stride": stride}

def candidate_configs(config):
    # Fixed public ordering; enrollment config is first. No score-based early exit.
    pool = [(config.block, config.stride)]
    for block in config.blind_blocks:
        for stride in (block, block // 2):
            if (block, stride) not in pool:
                pool.append((block, stride))
    return pool[:config.max_candidates]
