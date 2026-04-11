import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove, new_session
import io
import base64
import asyncio
import os
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
from skimage import color
from scipy.ndimage import binary_erosion

app = FastAPI(title="Smart Wardrobe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# TFLite Model Load
# ─────────────────────────────────────────
MODEL_PATH  = "fashion_model.tflite"
CLASS_NAMES = ["Bottomwear", "Dresses", "Topwear"]
IMG_SIZE    = 224

# ✅ FIX 2: Reduced from 900 → 512 (cuts rembg time ~50% since it's O(n²) in pixels)
MAX_BG_REMOVE_SIDE = 512

MAX_COLOR_PIXELS = 12000
COLOR_ANALYSIS_MAX_SIDE = 224
SHAPE_ANALYSIS_MAX_SIDE = 384
MAX_RESPONSE_IMAGE_SIDE = 1024

# Keep this below typical mobile HTTP timeouts so requests fail/continue quickly.
BG_REMOVE_TIMEOUT_SEC = 10
ENABLE_BG_REMOVAL = False

PIPELINE_VERSION = "2026-04-09-v5-speed"

print("Loading TFLite model...")
cpu_count = os.cpu_count() or 2
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=max(1, min(4, cpu_count)))
interpreter.allocate_tensors()
_interpreter_lock = threading.Lock()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite Model loaded!")

# ─────────────────────────────────────────
# ✅ FIX 1: Faster rembg session (silueta model — optimised for clothing/human silhouettes)
# Alternatives if silueta quality isn't enough: "isnet-general-use" or "birefnet-general"
# ─────────────────────────────────────────
print("Loading rembg session (silueta)...")
_rembg_session = new_session("silueta") if ENABLE_BG_REMOVAL else None
if ENABLE_BG_REMOVAL:
    print("rembg session loaded!")
else:
    print("Background removal disabled.")

# ─────────────────────────────────────────
# ✅ FIX 4: Pre-warm rembg — runs model once at startup so first real request is fast
# ─────────────────────────────────────────
print("Pre-warming rembg...")
if ENABLE_BG_REMOVAL:
    _dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    _dummy_buf = io.BytesIO()
    _dummy_img.save(_dummy_buf, format="PNG")
    remove(_dummy_buf.getvalue(), session=_rembg_session)
    del _dummy_img, _dummy_buf
    print("rembg warm!")
else:
    print("Skipped rembg warm-up.")

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def _get_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _downscale_for_speed(img: Image.Image, max_side: int = MAX_BG_REMOVE_SIDE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, _get_lanczos())


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _remove_bg_rgba_fast_sync(img: Image.Image, max_side: int = MAX_BG_REMOVE_SIDE) -> Image.Image:
    if not ENABLE_BG_REMOVAL or _rembg_session is None:
        return img.convert("RGBA")
    # ✅ FIX 1+2 combined: downscale to 512px max, then run with silueta session
    work_img = _downscale_for_speed(img, max_side)
    buff = io.BytesIO()
    work_img.save(buff, format="PNG")
    bg_removed = remove(buff.getvalue(), session=_rembg_session)   # ← session passed here
    return Image.open(io.BytesIO(bg_removed)).convert("RGBA")


async def _remove_bg_rgba_fast(img: Image.Image, timeout_sec: int = BG_REMOVE_TIMEOUT_SEC) -> Image.Image:
    if not ENABLE_BG_REMOVAL or _rembg_session is None:
        return img.convert("RGBA")
    # First try better quality, then retry with smaller size for speed.
    for max_side in (MAX_BG_REMOVE_SIDE, 384):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_remove_bg_rgba_fast_sync, img, max_side),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            continue
    raise TimeoutError(f"Background removal exceeded {timeout_sec}s")

def predict_category(img_array: np.ndarray) -> dict:
    with _interpreter_lock:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(preds))
    return {
        "category"   : CLASS_NAMES[pred_idx],
        "confidence" : round(float(preds[pred_idx]) * 100, 2),
        "all_scores" : {cls: round(float(p) * 100, 2) for cls, p in zip(CLASS_NAMES, preds)}
    }

# 🚀 JUGAD TO FIX CROP TOPS DETECTED AS SKIRTS
def _extract_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    transparent_ratio = float(np.mean(alpha < 250))
    opaque_ratio = float(np.mean(alpha > 250))
    has_meaningful_alpha = transparent_ratio > 0.01 and opaque_ratio > 0.01

    if has_meaningful_alpha and np.count_nonzero(alpha > 50) > 0:
        return alpha > 50

        # Fallback for RGB / fully-opaque RGBA inputs.
    if not ENABLE_BG_REMOVAL or _rembg_session is None:
        return np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
    img_rgba = remove(img, session=_rembg_session).convert("RGBA")
    arr = np.array(img_rgba)
    return arr[:, :, 3] > 50

def _crop_to_foreground(img_rgba: Image.Image, pad_ratio: float = 0.06) -> Image.Image:
    """
    Foreground alpha bbox ke around crop for more stable classification.
    """
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 50)
    if ys.size == 0 or xs.size == 0:
        return img_rgba

    h, w = alpha.shape
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    pad_y = int((y_max - y_min + 1) * pad_ratio)
    pad_x = int((x_max - x_min + 1) * pad_ratio)

    y0 = max(0, y_min - pad_y)
    y1 = min(h, y_max + pad_y + 1)
    x0 = max(0, x_min - pad_x)
    x1 = min(w, x_max + pad_x + 1)
    return img_rgba.crop((x0, y0, x1, y1))

def _is_pants_like(
    mask: np.ndarray,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    height: int,
    width: int,
) -> bool:
    """
    Jeans/pants shape detect karo:
    lower area mein 2 vertical legs + center gap hona chahiye.
    """
    if height < 40 or width < 20:
        return False

    lower_start = y_min + int(height * 0.48)
    lower_end = y_min + int(height * 0.97)
    if lower_end <= lower_start:
        return False

    roi = mask[lower_start:lower_end, x_min:x_max + 1]
    if roi.size == 0:
        return False

    min_leg_width = max(3, int(width * 0.10))
    min_gap = max(2, int(width * 0.05))

    rows_with_two_legs = 0
    valid_rows = 0

    for row in roi[::2]:
        cols = np.where(row > 0)[0]
        if cols.size < min_leg_width * 2:
            continue

        valid_rows += 1
        splits = np.where(np.diff(cols) > 1)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, cols.size - 1]

        runs = []
        for s_idx, e_idx in zip(starts, ends):
            run_start = cols[s_idx]
            run_end = cols[e_idx]
            run_w = run_end - run_start + 1
            if run_w >= min_leg_width:
                runs.append((run_start, run_end))

        if len(runs) < 2:
            continue

        runs = sorted(runs, key=lambda r: (r[1] - r[0]), reverse=True)[:2]
        left, right = sorted(runs, key=lambda r: r[0])
        gap = right[0] - left[1] - 1
        if gap >= min_gap:
            rows_with_two_legs += 1

    if valid_rows == 0:
        return False

    two_leg_ratio = rows_with_two_legs / valid_rows

    roi_main = _largest_component(roi)
    if np.count_nonzero(roi_main) == 0:
        return False

    area_threshold = roi.shape[0] * roi.shape[1] * 0.04
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(roi.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    strong_components = int(np.sum(areas >= area_threshold))

    return two_leg_ratio > 0.28 and strong_components >= 2

def _has_bottomwear_profile(mask: np.ndarray, y_min: int, y_max: int, x_min: int, x_max: int) -> bool:
    """
    Fallback profile: lower half mein center dip/gap bottomwear ka strong signal hai.
    """
    roi = mask[y_min:y_max + 1, x_min:x_max + 1]
    if roi.size == 0:
        return False

    h, w = roi.shape
    if h < 40 or w < 20:
        return False

    lower = roi[int(h * 0.45):, :]
    if lower.size == 0:
        return False

    lower_fill_ratio = float(np.mean(lower))

    c0, c1 = int(w * 0.45), int(w * 0.55)
    l0, l1 = 0, int(w * 0.20)
    r0, r1 = int(w * 0.80), w
    if c1 <= c0 or l1 <= l0 or r1 <= r0:
        return False

    center_fill = float(np.mean(lower[:, c0:c1]))
    side_fill = float(np.mean(np.hstack([lower[:, l0:l1], lower[:, r0:r1]])))
    if side_fill <= 1e-6:
        return False

    center_to_side = center_fill / side_fill
    return lower_fill_ratio < 0.74 and center_to_side < 0.82

def refine_category_by_shape(img: Image.Image, ml_category: str) -> str:
    try:
        mask = _extract_mask(img)

        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return ml_category

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        height = y_max - y_min
        width = max(1, x_max - x_min)

        if height < 20:
            return ml_category

        mid_y = y_min + int(height * 0.4)

        top_half_mask = mask[y_min:mid_y, :]
        top_coords = np.column_stack(np.where(top_half_mask > 0))
        top_width = (
            top_coords[:, 1].max() - top_coords[:, 1].min()
            if top_coords.size > 0 else 0
        )

        bot_half_mask = mask[mid_y:y_max, :]
        bot_coords = np.column_stack(np.where(bot_half_mask > 0))
        bot_width = (
            bot_coords[:, 1].max() - bot_coords[:, 1].min()
            if bot_coords.size > 0 else 0
        )

        height_ratio = height / img.size[1]
        y_start_ratio = y_min / img.size[1]

        top_band_h = max(2, int(height * 0.18))
        bottom_band_h = max(2, int(height * 0.18))

        top_band = mask[y_min:y_min + top_band_h, :]
        bottom_band = mask[y_max - bottom_band_h:y_max, :]

        top_band_coords = np.column_stack(np.where(top_band > 0))
        bottom_band_coords = np.column_stack(np.where(bottom_band > 0))

        top_band_width = (
            top_band_coords[:, 1].max() - top_band_coords[:, 1].min()
            if top_band_coords.size > 0 else 0
        )
        bottom_band_width = (
            bottom_band_coords[:, 1].max() - bottom_band_coords[:, 1].min()
            if bottom_band_coords.size > 0 else 0
        )

        top_band_ratio = top_band_width / width
        bottom_band_ratio = bottom_band_width / width
        pants_like = _is_pants_like(mask, y_min, y_max, x_min, x_max, height, width)
        bottomwear_profile = _has_bottomwear_profile(mask, y_min, y_max, x_min, x_max)

        if ml_category == "Dresses" and (pants_like or bottomwear_profile):
            print("PANTS SHAPE DETECTED — overriding Dresses to Bottomwear")
            return "Bottomwear"
        if ml_category == "Topwear" and (pants_like or bottomwear_profile) and height_ratio > 0.45:
            print("PANTS SHAPE DETECTED — overriding Topwear to Bottomwear")
            return "Bottomwear"

        if ml_category == "Bottomwear":
            waist_to_ankle_ratio = bottom_band_width / max(1, top_band_width)

            is_jeans_shape = (
                top_band_ratio < 0.75
                and bottom_band_ratio > 0.6
                and waist_to_ankle_ratio > 1.1
            )

            if is_jeans_shape:
                print(f"WIDE LEG JEANS DETECTED — skipping dress override | waist:{top_band_ratio:.2f} hem:{bottom_band_ratio:.2f}")
                return "Bottomwear"

            if (
                height_ratio > 0.58
                and y_start_ratio < 0.25
                and top_width > bot_width * 0.6
                and top_band_ratio > 0.65
                and not pants_like
                and not is_jeans_shape
            ):
                print(f"DRESS DETECTED | H:{height_ratio:.2f}, TW:{top_width}, BW:{bot_width}")
                return "Dresses"

            if height_ratio < 0.5 and top_width > bot_width * 1.05:
                print(f"TOP DETECTED | TW:{top_width}, BW:{bot_width}")
                return "Topwear"

            if (
                height_ratio > 0.6
                and y_start_ratio < 0.25
                and bot_width > top_width * 1.2
                and top_band_ratio > 0.65
                and not pants_like
            ):
                print("FLARE DRESS DETECTED")
                return "Dresses"

    except Exception as e:
        print(f"Shape refinement error: {e}")

    return ml_category

# ─────────────────────────────────────────
# Jeans / Bottomwear Sub-Type Detection
# ─────────────────────────────────────────
def detect_bottomwear_type(img: Image.Image) -> str:
    """
    Shape analysis se bottomwear ka type detect karo.
    """
    try:
        img = _downscale_for_speed(img, SHAPE_ANALYSIS_MAX_SIDE)
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        transparent_ratio = float(np.mean(alpha < 250))
        opaque_ratio = float(np.mean(alpha > 250))
        has_meaningful_alpha = transparent_ratio > 0.01 and opaque_ratio > 0.01

        if has_meaningful_alpha and np.count_nonzero(alpha > 50) > 0:
            mask = alpha > 50
        else:
            if ENABLE_BG_REMOVAL and _rembg_session is not None:
                img_rgba = remove(img, session=_rembg_session).convert("RGBA")
                arr = np.array(img_rgba)
                mask = arr[:, :, 3] > 50
            else:
                # Fallback when BG removal is disabled: suppress likely bright/neutral background.
                rgb_norm_full = arr[:, :, :3].astype(np.float32) / 255.0
                hsv_full = color.rgb2hsv(rgb_norm_full)
                bg_like = (hsv_full[:, :, 1] < 0.12) & (hsv_full[:, :, 2] > 0.88)
                mask = ~bg_like

                # For very light garments, avoid full-image fallback (it biases to Wide Leg).
                if np.count_nonzero(mask) < 100:
                    gray = np.mean(arr[:, :, :3].astype(np.float32), axis=2) / 255.0
                    mask = gray < 0.96
                    if np.count_nonzero(mask) < 100:
                        return "Unknown"

        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return "Unknown"

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Keep only the main connected foreground to remove edge/noise blobs.
        roi_full = mask[y_min:y_max + 1, x_min:x_max + 1]
        roi = _largest_component(roi_full)

        total_height = roi.shape[0]

        if total_height < 30:
            return "Unknown"

        def zone_width(start_pct, end_pct):
            y_start = int(total_height * start_pct)
            y_end = int(total_height * end_pct)
            if y_end <= y_start:
                return 0.0

            zone_mask = roi[y_start:y_end, :]
            spans = []
            for row in zone_mask[::2]:
                cols = np.where(row > 0)[0]
                if cols.size < 4:
                    continue
                splits = np.where(np.diff(cols) > 1)[0]
                starts = np.r_[0, splits + 1]
                ends = np.r_[splits, cols.size - 1]
                run_widths = [(cols[e] - cols[s] + 1) for s, e in zip(starts, ends)]
                if run_widths:
                    spans.append(max(run_widths))
            if len(spans) < 3:
                return 0.0
            return float(np.median(spans))

        hip_w = zone_width(0.00, 0.25)
        thigh_w = zone_width(0.25, 0.50)
        knee_w = zone_width(0.50, 0.75)
        ankle_w = zone_width(0.75, 1.00)

        print(f"Widths -> Hip:{hip_w} Thigh:{thigh_w} Knee:{knee_w} Ankle:{ankle_w}")

        if hip_w <= 1e-6:
            return "Unknown"

        knee_ratio = knee_w / hip_w
        ankle_ratio = ankle_w / hip_w
        thigh_ratio = thigh_w / hip_w

        print(f"Ratios -> Thigh:{thigh_ratio:.2f} Knee:{knee_ratio:.2f} Ankle:{ankle_ratio:.2f}")

        if abs(thigh_ratio - knee_ratio) < 0.08 and abs(knee_ratio - ankle_ratio) < 0.08 and 0.62 <= ankle_ratio <= 1.05:
            return "Straight Fit"

        if ankle_ratio < 0.45 and knee_ratio < 0.55:
            return "Skinny"

        if ankle_ratio < 0.55:
            return "Slim Fit"

        if thigh_ratio > 0.75 and ankle_ratio < 0.50:
            return "Tapered / Jogger"

        if ankle_w > knee_w * 1.18 and ankle_ratio > 0.62:
            return "Flared / Bootcut"

        if ankle_w > knee_w * 1.08 and ankle_ratio > 0.50:
            return "Bootcut"

        # Wide leg should have broad knee+ankle, not just one broad ankle value.
        if knee_ratio > 0.86 and ankle_ratio > 0.90 and abs(ankle_ratio - knee_ratio) < 0.18 and thigh_ratio > 0.78:
            return "Wide Leg"

        return "Regular Fit"

    except Exception as e:
        print(f"Bottomwear type detection error: {e}")
        return "Unknown"
def detect_color(img: Image.Image) -> str:
    try:
        img = _downscale_for_speed(img, COLOR_ANALYSIS_MAX_SIDE)
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]

        transparent_ratio = float(np.mean(alpha < 250))
        opaque_ratio = float(np.mean(alpha > 250))
        has_meaningful_alpha = transparent_ratio > 0.01 and opaque_ratio > 0.01

        if has_meaningful_alpha:
            mask = alpha > 180
        else:
            # When background removal is disabled, alpha is fully opaque.
            # Remove likely studio/background pixels (bright + low saturation).
            hsv_full = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
            sat = hsv_full[:, :, 1].astype(np.float32) / 255.0
            val = hsv_full[:, :, 2].astype(np.float32) / 255.0
            bg_like = (sat < 0.12) & (val > 0.88)
            mask = ~bg_like

            # If heuristic removed too much (e.g., white/gray garment), fall back.
            if np.count_nonzero(mask) < 100:
                mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)

        if np.count_nonzero(mask) < 100:
            return "Unknown"

        eroded = binary_erosion(mask, iterations=3)
        working_mask = eroded if np.count_nonzero(eroded) > 100 else mask

        pixels = arr[working_mask][:, :3].astype(np.float32)

        if len(pixels) > MAX_COLOR_PIXELS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pixels), size=MAX_COLOR_PIXELS, replace=False)
            pixels = pixels[idx]

        if len(pixels) < 50:
            return "Unknown"

        pixels_u8 = pixels.astype(np.uint8).reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(pixels_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)

        hue = (hsv_all[:, 0] * 2.0)
        sat = hsv_all[:, 1] / 255.0
        val = hsv_all[:, 2] / 255.0

        achromatic_mask = sat < 0.15
        chromatic_mask  = ~achromatic_mask

        chromatic_count = np.count_nonzero(chromatic_mask)
        achromatic_count = np.count_nonzero(achromatic_mask)
        total = len(pixels)

        if achromatic_count > chromatic_count:
            avg_val = float(np.median(val[achromatic_mask]))
            if avg_val < 0.20:
                return "Black"
            elif avg_val < 0.40:
                return "Charcoal"
            elif avg_val < 0.65:
                return "Gray"
            elif avg_val < 0.85:
                return "Light Gray"
            else:
                return "White"

        c_hue = hue[chromatic_mask]
        c_sat = sat[chromatic_mask]
        c_val = val[chromatic_mask]

        weights = c_sat ** 2
        sorted_idx = np.argsort(c_hue)
        cumulative = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumulative, cumulative[-1] / 2)
        dominant_hue = float(c_hue[sorted_idx[median_idx]])
        avg_sat = float(np.median(c_sat))
        avg_val = float(np.median(c_val))

        def hue_to_color(h, s, v):
            if h < 12 or h >= 348:
                if s < 0.4:
                    return "Pink"
                if v < 0.4:
                    return "Maroon"
                return "Red"

            if 12 <= h < 30:
                if v < 0.35:
                    return "Brown"
                if s < 0.5:
                    return "Beige"
                return "Orange"

            if 30 <= h < 65:
                if v < 0.55:
                    return "Mustard"
                if s < 0.5:
                    return "Beige"
                return "Yellow"

            if 65 <= h < 85:
                if v < 0.5:
                    return "Olive"
                return "Yellow"

            if 85 <= h < 150:
                if v < 0.35:
                    return "Olive"
                if s < 0.35:
                    return "Sage"
                return "Green"

            if 150 <= h < 195:
                return "Teal" if "Teal" in _palette() else "Green"

            if 195 <= h < 255:
                if v < 0.35:
                    return "Navy"
                if s < 0.4:
                    return "Light Blue"
                return "Blue"

            if 255 <= h < 290:
                if s < 0.4:
                    return "Lavender"
                return "Purple"

            if 290 <= h < 330:
                if v < 0.45:
                    return "Burgundy"
                if s < 0.5:
                    return "Pink"
                return "Hot Pink"

            if 330 <= h < 348:
                if v < 0.4:
                    return "Burgundy"
                return "Pink"

            return "Other"

        def _palette():
            return ["Teal", "Light Blue"]

        detected = hue_to_color(dominant_hue, avg_sat, avg_val)

        hue_rad = np.deg2rad(c_hue)
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        circular_std = np.sqrt(-2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2)))
        circular_std_deg = np.rad2deg(circular_std)

        if circular_std_deg > 60 and chromatic_count / total > 0.4:
            return "Multicolor"

        return detected

    except Exception as e:
        print("Color error:", e)
        return "Unknown"

class ImageBase64Request(BaseModel):
    image: str

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Smart Wardrobe API Running! 👗"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "pipeline_version": PIPELINE_VERSION}

@app.post("/classify")
async def classify_with_bg_removal(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP allowed")
    try:
        img_bytes  = await file.read()
        img_input  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bg_removed = ENABLE_BG_REMOVAL

        if bg_removed:
            try:
                img_rgba  = await _remove_bg_rgba_fast(img_input)
                img_focus = _crop_to_foreground(img_rgba)
            except TimeoutError:
                # Fallback: continue classification without BG removal to avoid hard timeout failures.
                bg_removed = False
                img_focus = img_input.convert("RGBA")
        else:
            img_focus = img_input.convert("RGBA")

        img_rgb    = img_focus.convert("RGB")
        img_array  = preprocess_image(img_rgb)
        result     = predict_category(img_array)

        # Skip shape refinement when bg removal timed out because it may call rembg again.
        if bg_removed:
            refined = refine_category_by_shape(img_focus, result["category"])
        else:
            refined = result["category"]

        return {
            "success": True,
            "filename": file.filename,
            "category": refined,
            "confidence": result["confidence"],
            "bg_removed": bg_removed,
            "timeout_fallback": not bg_removed,
            "pipeline_version": PIPELINE_VERSION,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/raw")
async def classify_without_bg_removal(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP allowed")
    try:
        img_bytes = await file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = preprocess_image(img)
        result    = predict_category(img_array)
        return {"success": True, "filename": file.filename, **result, "bg_removed": False, "pipeline_version": PIPELINE_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-clothing")
async def process_clothing(request: ImageBase64Request):
    try:
        image_data = base64.b64decode(request.image)
        img_input  = Image.open(io.BytesIO(image_data)).convert("RGB")
        bg_removed = ENABLE_BG_REMOVAL
        if bg_removed:
            try:
                img_rgba  = await _remove_bg_rgba_fast(img_input)
                img_focus = _crop_to_foreground(img_rgba)
            except TimeoutError:
                # Fallback: skip bg removal if rembg still times out (e.g. very slow hardware)
                bg_removed = False
                img_focus  = img_input.convert("RGBA")
        else:
            img_focus = img_input.convert("RGBA")

        img_array = preprocess_image(img_focus.convert("RGB"))
        result = predict_category(img_array)

        # If bg removal timed out, avoid shape-refinement fallback that may call rembg again.
        if bg_removed:
            final_category = refine_category_by_shape(img_focus, result["category"])
        else:
            final_category = result["category"]
        
        loop = asyncio.get_event_loop()
        color_future = loop.run_in_executor(None, detect_color, img_focus)
        sub_type_future = None
        if final_category == "Bottomwear":
           sub_type_future = loop.run_in_executor(None, detect_bottomwear_type, img_focus)

        detected_color = await color_future
        sub_type = None
        if sub_type_future is not None:
            sub_type = await sub_type_future
            print(f"👖 Bottomwear sub-type: {sub_type}")

        # detected_color = detect_color(img_focus)

        # sub_type = None
        # if final_category == "Bottomwear":
        #     sub_type = detect_bottomwear_type(img_focus)
            
        return {
            "success"   : True,
            "category"  : final_category,
            "sub_type"  : sub_type,
            "confidence": result["confidence"] / 100,
            "color"     : detected_color,
            "all_scores": result["all_scores"],
            "bg_removed": bg_removed,
            "pipeline_version": PIPELINE_VERSION,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/remove-background")
async def remove_bg_base64(request: ImageBase64Request):
    try:
        image_data = base64.b64decode(request.image)
        img_input = Image.open(io.BytesIO(image_data)).convert("RGB")
        bg_removed = ENABLE_BG_REMOVAL
        if bg_removed:
            try:
                fg_img = await _remove_bg_rgba_fast(img_input)
            except TimeoutError:
                # Fallback: do not fail request when background removal is slow.
                bg_removed = False
                fg_img = img_input.convert("RGBA")
        else:
            fg_img = img_input.convert("RGBA")

        grey_bg = Image.new("RGBA", fg_img.size, (176, 176, 176, 255))
        composite = Image.alpha_composite(grey_bg, fg_img)

        final_img = _downscale_for_speed(composite.convert("RGB"), MAX_RESPONSE_IMAGE_SIDE)

        buffer = io.BytesIO()
        final_img.save(buffer, format="PNG")
        processed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "success": True,
            "processed_image": processed_base64,
            "bg_removed": bg_removed,
            "timeout_fallback": not bg_removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)




