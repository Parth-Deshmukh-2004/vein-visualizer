from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def remove_led_bloom(gray: np.ndarray):
    """
    Mask + inpaint the bright LED hotspot so it doesn't corrupt
    CLAHE / contrast stretching downstream.
    Returns: (bloom_free_gray, bloom_mask)
    """
    # Anything above 220 is almost certainly the LED hotspot
    _, bloom = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # Also grab the adaptive top-1% brightest pixels (handles dimmer LEDs)
    p99 = np.percentile(gray, 99)
    _, bloom2 = cv2.threshold(gray, max(180, int(p99 * 0.92)), 255, cv2.THRESH_BINARY)
    bloom = cv2.bitwise_or(bloom, bloom2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bloom = cv2.dilate(bloom, kernel, iterations=2)   # cover halo ring

    fixed = cv2.inpaint(gray, bloom, inpaintRadius=18, flags=cv2.INPAINT_TELEA)
    return fixed, bloom


def extract_skin_mask(bloom_free: np.ndarray, bloom_mask: np.ndarray):
    """
    Robust skin-region mask that works even with bright LED blobs.
    """
    blurred = cv2.GaussianBlur(bloom_free, (15, 15), 0)
    _, fg = cv2.threshold(blurred, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8)
    if num_labels < 2:
        return np.ones_like(bloom_free, dtype=np.uint8) * 255

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(bloom_free, dtype=np.uint8)
    mask[labels == largest] = 255

    # Gentle erode to pull mask edge inward (avoids bright boundary pixels)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.erode(mask, k_erode, iterations=2)

    # Soft feather
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def frangi_vesselness(gray_f32: np.ndarray,
                      sigmas=(1.0, 1.5, 2.0, 3.0, 4.5),
                      beta=0.5, c=15.0) -> np.ndarray:
    """
    Multi-scale Hessian vessel filter (Frangi 1998).
    Highlights dark tubular structures (veins appear dark after inversion).
    Input:  float32, 0-255, inverted so veins are bright ridges.
    Output: float32 vesselness map, normalised 0-255.
    """
    vesselness = np.zeros_like(gray_f32)

    for sigma in sigmas:
        sm = cv2.GaussianBlur(gray_f32, (0, 0), sigma)

        Ixx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3)

        # Eigenvalues of 2x2 Hessian at each pixel
        tmp = np.sqrt(((Ixx - Iyy) / 2.0) ** 2 + Ixy ** 2)
        l1 = (Ixx + Iyy) / 2.0 + tmp   # larger  eigenvalue
        l2 = (Ixx + Iyy) / 2.0 - tmp   # smaller eigenvalue

        eps = 1e-6
        Rb = np.abs(l1) / (np.abs(l2) + eps)   # anisotropy (low = tube-like)
        S  = np.sqrt(l1 ** 2 + l2 ** 2)         # structure magnitude

        v = (np.exp(-(Rb ** 2) / (2 * beta ** 2)) *
             (1 - np.exp(-(S  ** 2) / (2 * c    ** 2))))

        # Only bright ridges (l2 < 0 in inverted image = dark tubes in original)
        v[l2 > 0] = 0.0

        vesselness = np.maximum(vesselness, v)

    vmax = vesselness.max()
    if vmax > 0:
        vesselness = vesselness / vmax * 255.0
    return vesselness


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_vein_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full pipeline → returns BGR image ready for projection:
      • Black background  → projector OFF → no light on skin (transparent)
      • Bright green veins → projector ON  → coloured light marks veins only
    """
    # ── Step 1: Best IR channel ───────────────────────────────────────────────
    # 850 nm IR leaks most into the red channel (index 2 in BGR)
    red = img_bgr[:, :, 2].copy()

    # ── Step 2: Remove LED bloom ──────────────────────────────────────────────
    bloom_free, bloom_mask = remove_led_bloom(red)

    # ── Step 3: Skin mask on bloom-corrected image ────────────────────────────
    skin_mask = extract_skin_mask(bloom_free, bloom_mask)

    # ── Step 4: Apply skin mask ───────────────────────────────────────────────
    masked = cv2.bitwise_and(bloom_free, bloom_free, mask=skin_mask)

    # ── Step 5: CLAHE #1 — global contrast boost ──────────────────────────────
    clahe1 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    enhanced = clahe1.apply(masked)

    # ── Step 6: High-pass filter — isolate vein spatial frequencies ───────────
    low_pass  = cv2.GaussianBlur(enhanced, (61, 61), 0)
    high_pass = cv2.subtract(enhanced, low_pass)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # ── Step 7: Invert (veins are dark → make them bright ridges for Frangi) ──
    inverted = cv2.bitwise_not(high_pass)
    inverted = cv2.bitwise_and(inverted, inverted, mask=skin_mask)

    # ── Step 8: CLAHE #2 — sharpen local contrast on inverted map ────────────
    clahe2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    inv_enh = clahe2.apply(inverted)

    # ── Step 9: Frangi multi-scale vessel filter ──────────────────────────────
    vessel_f  = frangi_vesselness(inv_enh.astype(np.float32),
                                  sigmas=(1.0, 1.5, 2.0, 3.0, 4.5))
    vessel_u8 = vessel_f.astype(np.uint8)
    vessel_u8 = cv2.bitwise_and(vessel_u8, vessel_u8, mask=skin_mask)

    # ── Step 10: Adaptive threshold on vesselness map ────────────────────────
    vein_mask = cv2.adaptiveThreshold(
        vessel_u8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25, C=-2          # negative C retains more thin veins
    )
    vein_mask = cv2.bitwise_and(vein_mask, vein_mask, mask=skin_mask)

    # ── Step 11: Morphological cleanup ────────────────────────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed  = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, k_close)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    # ── Step 12: Connected-component size filter ──────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)
    solid = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 300:
            solid[labels == i] = 255

    # ── Step 13: Skeletonize → re-dilate for clean line width ─────────────────
    try:
        thinned = cv2.ximgproc.thinning(solid)
    except AttributeError:
        thinned = solid   # opencv-contrib not installed — skip gracefully

    k_line      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_veins = cv2.dilate(thinned, k_line, iterations=2)

    # ── Step 14: Gamma correction — lifts dim veins for projector ─────────────
    gamma = 0.55    # <1 brightens midtones; thin veins remain visible on skin
    lut = np.array([int((i / 255.0) ** gamma * 255)
                    for i in range(256)], dtype=np.uint8)
    final_veins = cv2.LUT(final_veins, lut)

    # ── Step 15: Build GREEN-ON-BLACK projection image ────────────────────────
    #   Black pixels  → projector emits no light → skin looks normal
    #   Green pixels  → projector highlights exactly where veins are
    h, w = final_veins.shape
    projection = np.zeros((h, w, 3), dtype=np.uint8)
    projection[:, :, 1] = final_veins                                          # G channel
    projection[:, :, 0] = (final_veins.astype(np.float32) * 0.18).astype(np.uint8)  # tiny B tint

    return projection   # BGR, black bg, green/cyan veins


# ═══════════════════════════════════════════════════════════════════
#  FLASK ROUTES  — same API surface, Pi code needs zero changes
# ═══════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def root():
    return {"status": "vein processor running",
            "version": "2.0-projection",
            "endpoints": ["/health", "/process"]}, 200


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}, 200


@app.route('/process', methods=['POST'])
def process():
    file_bytes = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Bad image", 400

    processed = process_vein_image(img)
    _, encoded = cv2.imencode('.jpg', processed,
                              [cv2.IMWRITE_JPEG_QUALITY, 92])
    return send_file(io.BytesIO(encoded.tobytes()), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)