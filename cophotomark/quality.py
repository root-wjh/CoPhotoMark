import math
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else 10 * math.log10(255**2 / mse)

def measure(original, marked, boxes, block_counts, config, raw_codewords):
    a = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    b = cv2.cvtColor(marked, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mask = np.zeros(a.shape, dtype=bool)
    roi_values = []
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
        roi_values.append(psnr(a[y1:y2, x1:x2], b[y1:y2, x1:x2]))
    _, ssim_map = structural_similarity(a, b, data_range=255, full=True)
    pixels = a.size
    return {"quality_channel": "Y_uint8", "psnr_full_y_db": psnr(a, b),
            "psnr_roi_union_y_db": psnr(a[mask], b[mask]) if mask.any() else None,
            "psnr_roi_mean_db": float(np.mean([v for v in roi_values if v is not None])) if any(v is not None for v in roi_values) else None,
            "ssim_full_y": float(ssim_map[3:-3, 3:-3].mean()),
            "ssim_roi_union_y": float(ssim_map[mask].mean()) if mask.any() else None,
            "ssim_roi_definition": "Full-image SSIM map averaged over enrolled ROI union; includes window context",
            "roi_union_pixels": int(mask.sum()), "image_pixels": pixels,
            "protocol_bits": len(boxes) * config.payload_bits,
            "distinct_codeword_bits": len(set(raw_codewords)) * config.payload_bits,
            "embedded_bit_instances": sum(block_counts) * config.payload_bits,
            "protocol_bits_per_image_pixel": len(boxes)*config.payload_bits/pixels,
            "instances_per_image_pixel": sum(block_counts)*config.payload_bits/pixels,
            "instances_per_roi_pixel": sum(block_counts)*config.payload_bits/max(1, int(mask.sum())),
            "zero_mse_representation": "PSNR null denotes positive infinity for identical pixels"}
