from __future__ import annotations
import math
import random
import cv2
import numpy as np
from .engine import crop
from .detector import iou
from .lsbmeta import strip_metadata_lsb

def resize(image, scale_x, scale_y=None):
    scale_y = scale_x if scale_y is None else scale_y
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w*scale_x)), max(1, round(h*scale_y))), interpolation=cv2.INTER_LINEAR)

def jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def legacy_random_crop(image, crop_ratio, rng):
    """Retain a fraction of each side, draw x then y, and resize back."""
    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    if not 0 < crop_ratio <= 1 or min(ch, cw) < 1:
        raise ValueError("Legacy crop_ratio must yield a nonempty crop within the image")
    if isinstance(rng, random.Random):
        left = rng.randint(0, w - cw)
        top = rng.randint(0, h - ch)
    else:
        left = int(rng.integers(0, w - cw + 1))
        top = int(rng.integers(0, h - ch + 1))
    return cv2.resize(image[top:top+ch, left:left+cw], (w, h), interpolation=cv2.INTER_LINEAR)

def transform(image, spec, rng):
    name = spec["name"]
    if name == "clean":
        return image.copy()
    if name == "metadata_strip":
        # Explicit pixel-channel erasure control, not EXIF removal. Whether
        # DCT evidence survives must be measured by the blind verifier.
        return strip_metadata_lsb(image)
    if name == "jpeg":
        return jpeg(image, spec["quality"])
    if name == "resize":
        h, w = image.shape[:2]
        scale = spec["scale"]
        size = (int(w * scale), int(h * scale))
        if min(size) < 1:
            raise ValueError("Resize scale yields an empty image")
        out = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == "stretch":
        return resize(image, spec["scale_x"], spec["scale_y"])
    if name == "rotate":
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w//2, h//2), spec["degrees"], 1)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if name == "blur":
        sigma = spec["sigma"]
        kernel = spec.get("kernel", 5)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    if name == "crop":
        if "removed_area" in spec or spec.get("restore_size") is False:
            raise ValueError("Obsolete area-based crop spec: use crop_ratio and restore_size=True for the legacy protocol")
        return legacy_random_crop(image, spec.get("crop_ratio", 0.965), rng)
    if name == "cropout":
        out = image.copy()
        h, w = out.shape[:2]
        fraction = float(spec["area_fraction"])
        fill = spec.get("fill", 127)
        if not 0 <= fraction <= 1 or not 0 <= fill <= 255:
            raise ValueError("Cropout requires area_fraction in [0,1] and fill in [0,255]")
        if fraction == 0:
            return out
        ratio = math.sqrt(fraction)
        ch, cw = max(1, round(h*ratio)), max(1, round(w*ratio))
        top, left = rng.integers(0, h-ch+1), rng.integers(0, w-cw+1)
        out[top:top+ch, left:left+cw] = fill
        return out
    if name == "format_conversion":
        extension = spec.get("extension", ".webp")
        ok, encoded = cv2.imencode(extension, image)
        if not ok:
            raise ValueError("Format conversion failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if name == "screenshot_simulation":
        # This is raster rendering simulation, not an actual display capture.
        return jpeg(resize(image, spec.get("scale", 0.75)), spec.get("quality", 85))
    if name == "platform_simulation":
        out = image.copy()
        for _ in range(spec.get("rounds", 3)):
            out = jpeg(resize(out, spec.get("scale_per_round", 0.9)), spec.get("quality", 80))
        return out
    raise ValueError(f"Unknown transform {name}")

def paste(out, box, patch):
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = cv2.resize(patch, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)

def exact_reorder_boxes(boxes, shape):
    """Translate entire ROIs to cyclic destination anchors without cropping.

    Unequal source dimensions are retained. Unsupported overlapping or
    off-canvas layouts are explicitly skipped, never silently truncated.
    """
    if len(boxes) < 2:
        raise ValueError("Operation requires at least two faces")
    h, w = shape[:2]
    moved = []
    for dest, source in zip(boxes, boxes[-1:] + boxes[:-1]):
        x, y = dest[:2]
        new_box = [x, y, x + source[2] - source[0], y + source[3] - source[1]]
        if new_box[2] > w or new_box[3] > h or any(iou(new_box, b) > 0 for b in moved):
            raise ValueError("Exact reorder cannot preserve whole ROIs in this layout (overlap or canvas bound)")
        moved.append(new_box)
    return moved

def face_attack(image, boxes, spec, rng, donor=None, donor_box=None, donor_original=None, original=None, peers=None):
    if not boxes:
        raise ValueError("Face operation requires current enrolled boxes")
    out = image.copy()
    target = boxes[int(spec.get("target_index", 0)) % len(boxes)]
    name = spec["name"]
    if name == "delete":
        x1, y1, x2, y2 = target
        out[y1:y2, x1:x2] = spec.get("fill", 127)
    elif name in {"duplicate", "reorder"}:
        if len(boxes) < 2:
            raise ValueError("Operation requires at least two faces")
        if name == "duplicate":
            paste(out, boxes[1], crop(image, boxes[0]))
        else:
            for dest, source in zip(boxes, boxes[-1:] + boxes[:-1]):
                paste(out, dest, crop(image, source))
    elif name == "reorder_exact":
        moved = exact_reorder_boxes(boxes, image.shape)
        patches = [crop(image, box).copy() for box in boxes]
        for x1, y1, x2, y2 in boxes:
            out[y1:y2, x1:x2] = spec.get("fill", 127)
        for dest, patch in zip(moved, [patches[-1]] + patches[:-1]):
            x1, y1, x2, y2 = dest
            out[y1:y2, x1:x2] = patch
    elif name == "boundary_tamper":
        x1, y1, x2, y2 = target
        width = max(1, round(min(x2-x1, y2-y1)*spec.get("border_fraction", 0.1)))
        cv2.rectangle(out, (x1, y1), (x2-1, y2-1), (127, 127, 127), thickness=width)
    elif name in {"replace", "copy_marked_face", "watermark_transplant", "insert_unmarked", "insert_marked"}:
        if donor is None or donor_box is None:
            raise ValueError("Donor image and donor face box required")
        if name.startswith("insert"):
            # Extend canvas instead of overwriting an existing protected face.
            patch = crop(donor, donor_box)
            width = max(64, patch.shape[1])
            canvas = np.full((out.shape[0], out.shape[1]+width, 3), 127, dtype=np.uint8)
            canvas[:, :out.shape[1]] = out
            size = min(out.shape[0], max(64, patch.shape[0]))
            paste(canvas, [out.shape[1], 0, out.shape[1]+width, size], patch)
            out = canvas
        elif name == "watermark_transplant":
            if donor_original is None or original is None:
                raise ValueError("Residual transplant needs both unmarked originals")
            residual = donor.astype(np.float32) - donor_original.astype(np.float32)
            patch = crop(residual, donor_box)
            dest = crop(original, target).astype(np.float32)
            patch = cv2.resize(patch, (dest.shape[1], dest.shape[0]), interpolation=cv2.INTER_LINEAR)
            paste(out, target, np.clip(dest + patch, 0, 255).astype(np.uint8))
        else:
            paste(out, target, crop(donor, donor_box))
    elif name == "collusion":
        if not peers or len(peers) < 2 or any(p.shape != image.shape for p in peers):
            raise ValueError("Collusion needs >=2 independently marked versions of the SAME host, equal shape")
        out = np.clip(np.rint(np.stack(peers).astype(np.float32).mean(axis=0)), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported face attack {name}")
    return out

def default_plan():
    specs = [{"name": "clean", "label": "genuine"}, {"name": "metadata_strip", "label": "genuine"}]
    specs += [{"name": "jpeg", "quality": q, "label": "genuine"} for q in (95, 85, 70, 50)]
    specs += [{"name": "resize", "scale": s, "label": "genuine"} for s in (0.5, 0.75, 0.9, 1.25)]
    specs += [{"name": "blur", "sigma": s, "kernel": 5, "label": "genuine"} for s in (0.5, 1.0, 2.0)]
    specs += [{"name": "rotate", "degrees": a, "label": "genuine"} for a in (-5, 5)]
    specs += [{"name": "stretch", "scale_x": 1.2, "scale_y": 0.8, "label": "genuine"},
              {"name": "crop", "crop_ratio": 0.965, "restore_size": True, "label": "stress"},
              {"name": "cropout", "area_fraction": 0.3, "fill": 127, "label": "stress"},
              {"name": "screenshot_simulation", "scale": 0.75, "quality": 85, "label": "genuine"},
              {"name": "platform_simulation", "rounds": 3, "scale_per_round": 0.9, "quality": 80, "label": "genuine"},
              {"name": "format_conversion", "extension": ".webp", "label": "genuine"}]
    return specs
