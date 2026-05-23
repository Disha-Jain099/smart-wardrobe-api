import numpy as np
from PIL import Image
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

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception as e:
        raise RuntimeError(
            "TFLite interpreter not available. Install `tensorflow-cpu==2.16.1` "
            "or `tflite-runtime` and redeploy."
        ) from e

app = FastAPI(title="Smart Wardrobe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLOTHING_MODEL_PATH = "fashion_model.tflite"
FOOTWEAR_MODEL_PATH = "footwear_classifier.tflite"
FOOTWEAR_LABELS_PATH = "labels.txt"
CLOTHING_CLASS_NAMES = ["Bottomwear", "Dresses", "Topwear"]
IMG_SIZE = 224
FOOTWEAR_CATEGORY_NAME = "Footwear"
FOOTWEAR_CONFIDENCE_THRESHOLD = 55.0

MAX_BG_REMOVE_SIDE = 512
MAX_COLOR_PIXELS = 12000
COLOR_ANALYSIS_MAX_SIDE = 224
SHAPE_ANALYSIS_MAX_SIDE = 384
MAX_RESPONSE_IMAGE_SIDE = 1024
BG_REMOVE_TIMEOUT_SEC = 10
ENABLE_BG_REMOVAL = os.getenv("ENABLE_BG_REMOVAL", "false").lower() == "true"
PIPELINE_VERSION = "2026-04-12-v6-footwear"

print("Loading TFLite models...")
cpu_count = os.cpu_count() or 2
_num_threads = max(1, min(4, cpu_count))

try:
    clothing_interpreter = Interpreter(model_path=CLOTHING_MODEL_PATH, num_threads=_num_threads)
    clothing_interpreter.allocate_tensors()
    _clothing_interpreter_lock = threading.Lock()
    clothing_input_details = clothing_interpreter.get_input_details()
    clothing_output_details = clothing_interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(
        f"Failed to load clothing model '{CLOTHING_MODEL_PATH}'. "
        "Model was likely exported with newer TFLite ops than current runtime supports."
    ) from e


def _load_footwear_labels(path: str) -> list[str]:
    labels = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Footwear labels file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                _, label = line.split(":", 1)
                line = label.strip()
            labels.append(line)
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


FOOTWEAR_CLASS_NAMES = _load_footwear_labels(FOOTWEAR_LABELS_PATH)
try:
    footwear_interpreter = Interpreter(model_path=FOOTWEAR_MODEL_PATH, num_threads=_num_threads)
    footwear_interpreter.allocate_tensors()
    _footwear_interpreter_lock = threading.Lock()
    footwear_input_details = footwear_interpreter.get_input_details()
    footwear_output_details = footwear_interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(
        f"Failed to load footwear model '{FOOTWEAR_MODEL_PATH}'. "
        "Model was likely exported with newer TFLite ops than current runtime supports."
    ) from e
print("TFLite models loaded!")

_rembg_session = None
_rembg_remove = None

if ENABLE_BG_REMOVAL:
    try:
        from rembg import remove as _rembg_remove_fn, new_session as _rembg_new_session

        _rembg_session = _rembg_new_session("u2netp")
        _rembg_remove = _rembg_remove_fn
        print("Background removal enabled with rembg.")
    except Exception as e:
        ENABLE_BG_REMOVAL = False
        print(f"Background removal disabled (rembg unavailable): {e}")
else:
    print("Background removal disabled by config.")

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
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

def _fallback_remove_background(img_rgb: Image.Image) -> Image.Image:
    arr = np.array(img_rgb.convert("RGB"))
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return img_rgb.convert("RGBA")
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, int(w * 0.05)), max(1, int(h * 0.05)), max(2, int(w * 0.90)), max(2, int(h * 0.90)))
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = np.full((h, w), 255, dtype=np.uint8)
    rgba = np.dstack([arr, fg])
    return Image.fromarray(rgba, "RGBA")


def remove_background_image(img_rgb: Image.Image) -> tuple[Image.Image, bool]:
    if ENABLE_BG_REMOVAL and _rembg_remove is not None and _rembg_session is not None:
        try:
            img_small = _downscale_for_speed(img_rgb, MAX_BG_REMOVE_SIDE)
            out = _rembg_remove(img_small, session=_rembg_session)
            if isinstance(out, bytes):
                rgba = Image.open(io.BytesIO(out)).convert("RGBA")
            else:
                rgba = out.convert("RGBA")
            return rgba, True
        except Exception as e:
            print(f"rembg error, using fallback: {e}")
    return _fallback_remove_background(img_rgb), False


def predict_category(img_array: np.ndarray) -> dict:
    with _clothing_interpreter_lock:
        clothing_interpreter.set_tensor(clothing_input_details[0]['index'], img_array)
        clothing_interpreter.invoke()
        preds = clothing_interpreter.get_tensor(clothing_output_details[0]['index'])[0]
    pred_idx = int(np.argmax(preds))
    return {
        "category"   : CLOTHING_CLASS_NAMES[pred_idx],
        "confidence" : round(float(preds[pred_idx]) * 100, 2),
        "all_scores" : {cls: round(float(p) * 100, 2) for cls, p in zip(CLOTHING_CLASS_NAMES, preds)}
    }


def predict_footwear_category(img_array: np.ndarray) -> dict:
    with _footwear_interpreter_lock:
        footwear_interpreter.set_tensor(footwear_input_details[0]['index'], img_array)
        footwear_interpreter.invoke()
        preds = footwear_interpreter.get_tensor(footwear_output_details[0]['index'])[0]
    pred_idx = int(np.argmax(preds))
    return {
        "category": FOOTWEAR_CLASS_NAMES[pred_idx],
        "confidence": round(float(preds[pred_idx]) * 100, 2),
        "all_scores": {cls: round(float(p) * 100, 2) for cls, p in zip(FOOTWEAR_CLASS_NAMES, preds)},
    }


FOOTWEAR_STRONG_CONF   = 85.0
CLOTHING_OVERRIDE_CONF = 85.0

def resolve_final_category(img: Image.Image, clothing_result: dict, footwear_result: dict) -> tuple[str, float]:
    clothing_conf = clothing_result["confidence"]
    footwear_conf = footwear_result["confidence"]
    
    # Image ki shape check karein
    w, h = img.size
    aspect_ratio = h / w 

    # DEBUG logs (Taki aap terminal mein dekh sakein kya ho raha hai)
    print(f"DEBUG: Footwear {footwear_conf}% | Clothing {clothing_conf}% | Aspect {aspect_ratio:.2f}")

    # RULE 1: Agar Footwear model confident hai (>60%) AUR image tall nahi hai, 
    # toh wo pakka Footwear hi hai.
    if footwear_conf > 60.0 and aspect_ratio < 1.2:
        return FOOTWEAR_CATEGORY_NAME, footwear_conf

    # RULE 2: Agar model confuse hai, toh shape refinement chalao
    refined_cat = refine_category_by_shape(img, clothing_result["category"])

    # RULE 3: Agar image kafi lambi hai (Pants/Dress), toh footwear ko ignore karo
    if aspect_ratio > 1.5 and footwear_conf < 90.0:
        return refined_cat, clothing_conf

    # Default logic (Confidence based)
    if footwear_conf > clothing_conf + 5.0:
        return FOOTWEAR_CATEGORY_NAME, footwear_conf
        
    return refined_cat, clothing_conf

def _extract_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    transparent_ratio = float(np.mean(alpha < 250))
    opaque_ratio = float(np.mean(alpha > 250))
    has_meaningful_alpha = transparent_ratio > 0.01 and opaque_ratio > 0.01
    if has_meaningful_alpha and np.count_nonzero(alpha > 50) > 0:
        return alpha > 50
    return np.ones((arr.shape[0], arr.shape[1]), dtype=bool)

def _crop_to_foreground(img_rgba: Image.Image, pad_ratio: float = 0.06) -> Image.Image:
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

def _is_pants_like(mask, y_min, y_max, x_min, x_max, height, width):
    """
    Checks if the mask shape resembles pants (connected at top, split at bottom).
    """
    # 1. Minimum size check - bahut choti cheez pants nahi ho sakti
    if height < 50 or width < 30:
        return False

    # --- STEP 1: WAIST CONNECTIVITY CHECK (Fix for Footwear confusion) ---
    # Hum top 20% area check karenge. Agar ye pants hain, toh waist ek single piece honi chahiye.
    # Agar upar hi do alag blobs hain, toh wo pakka footwear ya kuch aur hai.
    waist_h = int(height * 0.2)
    waist_zone = mask[y_min:y_min + waist_h, x_min:x_max + 1].astype(np.uint8)
    
    # connectedComponents returns (num_labels, labels)
    num_labels, _ = cv2.connectedComponents(waist_zone, connectivity=8)
    
    # num_labels mein background bhi counted hota hai (label 0). 
    # Agar 1 se zyada object pieces hain (num_labels > 2), toh ye pants nahi hain.
    if num_labels > 2:
        return False

    # --- STEP 2: LEG DETECTION (Lower half check) ---
    # Hum niche ke 50% area mein do "legs" dhundhenge
    lower_start = y_min + int(height * 0.50)
    lower_end = y_min + int(height * 0.95)
    
    if lower_end <= lower_start:
        return False
        
    roi = mask[lower_start:lower_end, x_min:x_max + 1]
    if roi.size == 0:
        return False

    # Parameters for leg detection
    min_leg_width = max(4, int(width * 0.12))
    min_gap = max(3, int(width * 0.08))
    
    rows_with_two_legs = 0
    valid_rows = 0
    
    # Har dusri row check karo speed ke liye
    for row in roi[::3]:
        cols = np.where(row > 0)[0]
        if cols.size < (min_leg_width * 2):
            continue
            
        valid_rows += 1
        # Find gaps between pixels in the row
        splits = np.where(np.diff(cols) > 1)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, cols.size - 1]
        
        runs = []
        for s_idx, e_idx in zip(starts, ends):
            run_w = cols[e_idx] - cols[s_idx] + 1
            if run_w >= min_leg_width:
                runs.append((cols[s_idx], cols[e_idx]))
        
        if len(runs) >= 2:
            # Sort by width and take two largest
            runs = sorted(runs, key=lambda r: (r[1] - r[0]), reverse=True)[:2]
            # Sort by position to find gap
            left, right = sorted(runs, key=lambda r: r[0])
            gap = right[0] - left[1]
            if gap >= min_gap:
                rows_with_two_legs += 1

    if valid_rows == 0:
        return False

    # Ratio check: Kitni rows mein do legs dikhi?
    two_leg_ratio = rows_with_two_legs / valid_rows
    
    # Final check: Components in the ROI
    num_labels_roi, _, stats, _ = cv2.connectedComponentsWithStats(roi.astype(np.uint8), connectivity=8)
    # Area threshold to ignore tiny noise
    area_thresh = roi.size * 0.05
    strong_components = np.sum(stats[1:, cv2.CC_STAT_AREA] >= area_thresh)

    # Logic: Agar two_leg_ratio accha hai AUR niche do alag blobs hain
    return (two_leg_ratio > 0.30) and (strong_components >= 2)

def _has_bottomwear_profile(mask, y_min, y_max, x_min, x_max):
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
        top_width = (top_coords[:, 1].max() - top_coords[:, 1].min() if top_coords.size > 0 else 0)
        bot_half_mask = mask[mid_y:y_max, :]
        bot_coords = np.column_stack(np.where(bot_half_mask > 0))
        bot_width = (bot_coords[:, 1].max() - bot_coords[:, 1].min() if bot_coords.size > 0 else 0)
        height_ratio = height / img.size[1]
        y_start_ratio = y_min / img.size[1]
        top_band_h = max(2, int(height * 0.18))
        bottom_band_h = max(2, int(height * 0.18))
        top_band = mask[y_min:y_min + top_band_h, :]
        bottom_band = mask[y_max - bottom_band_h:y_max, :]
        top_band_coords = np.column_stack(np.where(top_band > 0))
        bottom_band_coords = np.column_stack(np.where(bottom_band > 0))
        top_band_width = (top_band_coords[:, 1].max() - top_band_coords[:, 1].min() if top_band_coords.size > 0 else 0)
        bottom_band_width = (bottom_band_coords[:, 1].max() - bottom_band_coords[:, 1].min() if bottom_band_coords.size > 0 else 0)
        top_band_ratio = top_band_width / width
        bottom_band_ratio = bottom_band_width / width
        pants_like = _is_pants_like(mask, y_min, y_max, x_min, x_max, height, width)
        bottomwear_profile = _has_bottomwear_profile(mask, y_min, y_max, x_min, x_max)
        if ml_category == "Dresses" and (pants_like or bottomwear_profile):
            return "Bottomwear"
        if ml_category == "Topwear" and (pants_like or bottomwear_profile) and height_ratio > 0.45:
            return "Bottomwear"
        if ml_category == "Bottomwear":
            waist_to_ankle_ratio = bottom_band_width / max(1, top_band_width)
            is_jeans_shape = (top_band_ratio < 0.75 and bottom_band_ratio > 0.6 and waist_to_ankle_ratio > 1.1)
            if is_jeans_shape:
                return "Bottomwear"
            if (height_ratio > 0.58 and y_start_ratio < 0.25 and top_width > bot_width * 0.6 and top_band_ratio > 0.65 and not pants_like and not is_jeans_shape):
                return "Dresses"
            if height_ratio < 0.5 and top_width > bot_width * 1.05:
                return "Topwear"
            if (height_ratio > 0.6 and y_start_ratio < 0.25 and bot_width > top_width * 1.2 and top_band_ratio > 0.65 and not pants_like):
                return "Dresses"
    except Exception as e:
        print(f"Shape refinement error: {e}")
    return ml_category

def detect_bottomwear_type(img: Image.Image) -> str:
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
            rgb_norm_full = arr[:, :, :3].astype(np.float32) / 255.0
            hsv_full = cv2.cvtColor((rgb_norm_full * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv_full[:, :, 0] = hsv_full[:, :, 0] / 179.0
            hsv_full[:, :, 1] = hsv_full[:, :, 1] / 255.0
            hsv_full[:, :, 2] = hsv_full[:, :, 2] / 255.0
            bg_like = (hsv_full[:, :, 1] < 0.12) & (hsv_full[:, :, 2] > 0.88)
            mask = ~bg_like
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
        if hip_w <= 1e-6:
            return "Unknown"
        knee_ratio = knee_w / hip_w
        ankle_ratio = ankle_w / hip_w
        thigh_ratio = thigh_w / hip_w
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
        if knee_ratio > 0.86 and ankle_ratio > 0.90 and abs(ankle_ratio - knee_ratio) < 0.18 and thigh_ratio > 0.78:
            return "Wide Leg"
        return "Regular Fit"
    except Exception as e:
        print(f"Bottomwear type detection error: {e}")
        return "Unknown"

def detect_color(img: Image.Image) -> str:
    """
    Improved color detection with center-weighting to ignore background noise.
    """
    try:
        # 1. Downscale for performance
        img = _downscale_for_speed(img, COLOR_ANALYSIS_MAX_SIDE)
        w, h = img.size
        arr = np.array(img.convert("RGBA"))
        
        # 2. Smart Masking Logic
        alpha = arr[:, :, 3]
        transparent_ratio = float(np.mean(alpha < 250))
        
        if transparent_ratio > 0.05:
            # If we have a transparent background, use alpha
            mask = alpha > 180
        else:
            # If no transparency, focus on the CENTER (middle 60% area)
            # This ignores walls/floors that usually occupy the edges
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            mask = np.zeros((h, w), dtype=bool)
            mask[margin_y:h-margin_y, margin_x:w-margin_x] = True
            
            # Refine: Ignore very bright (white-ish) backgrounds within that center
            hsv_center = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
            sat = hsv_center[:, :, 1].astype(np.float32) / 255.0
            val = hsv_center[:, :, 2].astype(np.float32) / 255.0
            not_bg = ~((sat < 0.1) & (val > 0.9)) # Ignore high brightness + low saturation
            mask = mask & not_bg

        if np.count_nonzero(mask) < 50:
            return "Unknown"

        # 3. Erosion to remove "edge bleed" from background
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)
        working_mask = eroded if np.count_nonzero(eroded) > 100 else mask
        
        # 4. Pixel Extraction & Sampling
        pixels = arr[working_mask][:, :3].astype(np.float32)
        if len(pixels) > MAX_COLOR_PIXELS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pixels), size=MAX_COLOR_PIXELS, replace=False)
            pixels = pixels[idx]

        # 5. Convert to HSV for robust analysis
        pixels_u8 = pixels.astype(np.uint8).reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(pixels_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        
        hue = hsv_all[:, 0] * 2.0  # Scale to 0-360
        sat = hsv_all[:, 1] / 255.0
        val = hsv_all[:, 2] / 255.0

        # 6. Separate Achromatic (Black/White/Gray) from Chromatic
        achromatic_mask = (sat < 0.15) | (val < 0.15)
        chromatic_mask = ~achromatic_mask
        
        chrom_count = np.count_nonzero(chromatic_mask)
        achrom_count = np.count_nonzero(achromatic_mask)

        # 7. Decision Logic
        if achrom_count > chrom_count * 1.5:
            avg_val = float(np.median(val[achromatic_mask]))
            if avg_val < 0.20: return "Black"
            if avg_val < 0.45: return "Charcoal"
            if avg_val < 0.75: return "Gray"
            if avg_val < 0.90: return "Light Gray"
            return "White"

        # Handle Chromatic Colors
        c_hue = hue[chromatic_mask]
        c_sat = sat[chromatic_mask]
        c_val = val[chromatic_mask]
        
        # Weighted Median Hue
        sorted_idx = np.argsort(c_hue)
        weights = (c_sat ** 2) * c_val # Prioritize vivid/bright pixels
        cumulative = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumulative, cumulative[-1] / 2)
        dom_hue = float(c_hue[sorted_idx[median_idx]])
        
        avg_sat = float(np.median(c_sat))
        avg_val = float(np.median(c_val))

        # 8. Hue Mapping
        def hue_to_name(h, s, v):
            if h < 15 or h >= 345:
                return "Pink" if s < 0.4 else ("Maroon" if v < 0.4 else "Red")
            if 15 <= h < 45:
                return "Brown" if v < 0.4 else ("Beige" if s < 0.4 else "Orange")
            if 45 <= h < 70:
                return "Mustard" if v < 0.6 else ("Beige" if s < 0.3 else "Yellow")
            if 70 <= h < 160:
                return "Olive" if v < 0.4 else ("Sage" if s < 0.4 else "Green")
            if 160 <= h < 200:
                return "Teal" if v < 0.5 else "Cyan"
            if 200 <= h < 260:
                return "Navy" if v < 0.4 else ("Light Blue" if s < 0.4 else "Blue")
            if 260 <= h < 300:
                return "Purple"
            if 300 <= h < 345:
                return "Burgundy" if v < 0.4 else "Pink"
            return "Other"

        detected = hue_to_name(dom_hue, avg_sat, avg_val)

        # 9. Multicolor Check (Circular Variance)
        hue_rad = np.deg2rad(c_hue)
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        circular_std = np.rad2deg(np.sqrt(-2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2))))
        
        if circular_std > 55 and (chrom_count / len(pixels)) > 0.4:
            return "Multicolor"

        return detected
    except Exception as e:
        print(f"Color Analysis Error: {e}")
        return "Unknown"

class ImageBase64Request(BaseModel):
    image: str

@app.get("/")
def home():
    return {"message": "Smart Wardrobe API Running!"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "clothing": True,
            "footwear": True,
        },
        "pipeline_version": PIPELINE_VERSION,
    }

@app.post("/classify")
async def classify_with_bg_removal(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP allowed")
    try:
        img_bytes = await file.read()
        img_input = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_focus = img_input.convert("RGBA")
        img_array = preprocess_image(img_focus.convert("RGB"))
        clothing_result = predict_category(img_array)
        footwear_result = predict_footwear_category(img_array)
        final_category, final_confidence = resolve_final_category(img_focus, clothing_result, footwear_result)
        return {
            "success": True,
            "filename": file.filename,
            "category": final_category,
            "confidence": final_confidence,
            "clothing_category": clothing_result["category"],
            "clothing_confidence": clothing_result["confidence"],
            "footwear_category": footwear_result["category"],
            "footwear_confidence": footwear_result["confidence"],
            "bg_removed": False,
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
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = preprocess_image(img)
        clothing_result = predict_category(img_array)
        footwear_result = predict_footwear_category(img_array)
        final_category, final_confidence = resolve_final_category(img, clothing_result, footwear_result)
        return {
            "success": True,
            "filename": file.filename,
            "category": final_category,
            "confidence": final_confidence,
            "clothing": clothing_result,
            "footwear": footwear_result,
            "bg_removed": False,
            "pipeline_version": PIPELINE_VERSION,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-clothing")
async def process_clothing(request: ImageBase64Request):
    try:
        image_data = base64.b64decode(request.image)
        img_input = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_focus = img_input.convert("RGBA")
        img_array = preprocess_image(img_focus.convert("RGB"))
        clothing_result = predict_category(img_array)
        footwear_result = predict_footwear_category(img_array)
        final_category, final_confidence = resolve_final_category(img_focus, clothing_result, footwear_result)
        loop = asyncio.get_event_loop()
        color_future = loop.run_in_executor(None, detect_color, img_focus)
        sub_type_future = None
        footwear_type = None
        if final_category == "Bottomwear":
            sub_type_future = loop.run_in_executor(None, detect_bottomwear_type, img_focus)
        elif final_category == FOOTWEAR_CATEGORY_NAME:
            footwear_type = footwear_result["category"]
        detected_color = await color_future
        sub_type = None
        if sub_type_future is not None:
            sub_type = await sub_type_future
        return {
            "success"   : True,
            "category"  : final_category,
            "sub_type"  : sub_type,
            "footwear_type": footwear_type,
            "confidence": round(final_confidence / 100.0, 4),
            "color"     : detected_color,
            "all_scores": clothing_result["all_scores"],
            "clothing": clothing_result,
            "footwear": footwear_result,
            "bg_removed": False,
            "pipeline_version": PIPELINE_VERSION,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/remove-background")
async def remove_bg_base64(request: ImageBase64Request):
    try:
        image_data = base64.b64decode(request.image)
        img_input = Image.open(io.BytesIO(image_data)).convert("RGB")
        fg_img, used_rembg = remove_background_image(img_input)
        grey_bg = Image.new("RGBA", fg_img.size, (176, 176, 176, 255))
        composite = Image.alpha_composite(grey_bg, fg_img)
        final_img = _downscale_for_speed(composite.convert("RGB"), MAX_RESPONSE_IMAGE_SIDE)
        buffer = io.BytesIO()
        final_img.save(buffer, format="PNG")
        processed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {
            "success": True,
            "processed_image": processed_base64,
            "bg_removed": True,
            "bg_engine": "rembg" if used_rembg else "opencv_fallback",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
