#!/usr/bin/env python3
"""
Vein Viewer – Raspberry Pi Zero 2W
  • Captures IR frames from ArduCam v2
  • Sends to Hugging Face for vein processing
  • Projects the result back onto skin via HDMI display / mini-projector
  
Install deps on Pi:
  pip install picamera2 opencv-python-headless flask requests pygame
"""

import cv2
import numpy as np
import requests
import threading
import time
from flask import Flask, Response, jsonify, request, render_template_string
from picamera2 import Picamera2
import pygame
import sys

app = Flask(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HF_URL = "https://Parttthhhhhhhhh-vein-processor.hf.space/process"

# Set to the resolution your projector/display outputs at
PROJECTOR_W = 1280
PROJECTOR_H = 720

settings = {
    'exposure':      3500,   # microseconds — raise if image is too dark
    'gain':          1.5,    # analogue gain — keep low to reduce noise
    'clahe_clip':    6.0,
    # Projection transform: if projector image is offset/rotated vs camera FOV
    # tweak these to align the green overlay with the patient's hand
    'proj_scale':    1.0,   # zoom factor (1.0 = fill projector height)
    'proj_offset_x': 0,     # horizontal pixel shift
    'proj_offset_y': 0,     # vertical pixel shift
    'proj_rotate':   0,     # degrees CW: 0 / 90 / 180 / 270
    'brightness':    1.0,   # projector image brightness multiplier (0.5–2.0)
}

state = {
    'live_jpeg':      None,
    'processed_jpeg': None,
    'is_processing':  False,
}

lock = threading.Lock()
picam2 = None

# ── CAMERA INIT ───────────────────────────────────────────────────────────────
def init_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (820, 616), "format": "RGB888"},
        controls={
            "NoiseReductionMode": 0,
            "Sharpness":          15.0,
            "Saturation":         0.0,
            "Brightness":         -0.1,
            "Contrast":           1.8,
        }
    )
    picam2.configure(config)
    picam2.start()
    _apply_camera_controls()
    print("✓ Camera ready")

def _apply_camera_controls():
    if picam2:
        # AfMode and LensPosition are NOT available on fixed-focus modules
        # (ArduCam v2 / IMX219 without motorised lens) — omit them entirely
        picam2.set_controls({
            "AeEnable":     False,
            "AwbEnable":    False,
            "ExposureTime": int(settings['exposure']),
            "AnalogueGain": float(settings['gain']),
        })

# ── BACKGROUND THREADS ────────────────────────────────────────────────────────
def capture_loop():
    while True:
        try:
            raw = picam2.capture_array()
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with lock:
                state['live_jpeg'] = jpeg.tobytes()
        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(1)

def cloud_sync_worker():
    while True:
        frame = None
        with lock:
            if state['live_jpeg'] and not state['is_processing']:
                frame = state['live_jpeg']
        if frame:
            threading.Thread(target=_process_on_cloud, args=(frame,),
                             daemon=True).start()
        time.sleep(0.4)   # ~2.5 cloud requests/sec max

def _process_on_cloud(image_bytes: bytes):
    with lock:
        state['is_processing'] = True
    try:
        r = requests.post(
            HF_URL,
            data=image_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=10
        )
        if r.status_code == 200:
            with lock:
                state['processed_jpeg'] = r.content
    except Exception as e:
        print(f"Cloud error: {e}")
    finally:
        with lock:
            state['is_processing'] = False

# ── PROJECTOR LOOP (runs on main thread so pygame has display ownership) ──────
def build_projection_surface(jpeg_bytes: bytes) -> pygame.Surface:
    """
    Decode the green-on-black JPEG from HF, apply user transforms,
    scale to fit projector resolution, return a pygame.Surface.
    """
    arr  = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    # Apply brightness multiplier
    b = settings['brightness']
    if b != 1.0:
        bgr = np.clip(bgr.astype(np.float32) * b, 0, 255).astype(np.uint8)

    # Optional rotation
    rot = int(settings['proj_rotate']) % 360
    if rot == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rot == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Scale to fit projector while keeping aspect ratio
    ih, iw = bgr.shape[:2]
    scale = min(PROJECTOR_W / iw, PROJECTOR_H / ih) * settings['proj_scale']
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre on black canvas + user offset
    canvas = np.zeros((PROJECTOR_H, PROJECTOR_W, 3), dtype=np.uint8)
    x0 = (PROJECTOR_W - new_w) // 2 + int(settings['proj_offset_x'])
    y0 = (PROJECTOR_H - new_h) // 2 + int(settings['proj_offset_y'])
    # Clip to canvas bounds
    x0 = max(0, min(x0, PROJECTOR_W - 1))
    y0 = max(0, min(y0, PROJECTOR_H - 1))
    x1 = min(x0 + new_w, PROJECTOR_W)
    y1 = min(y0 + new_h, PROJECTOR_H)
    canvas[y0:y1, x0:x1] = bgr[: y1 - y0, : x1 - x0]

    # BGR → RGB for pygame
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return surface

def projector_loop():
    """
    pygame window on the HDMI output.
    Set your Pi to output to the projector via HDMI; this window goes fullscreen.
    Press ESC or Q to quit.
    """
    pygame.init()
    # NOFRAME + FULLSCREEN → borderless full-screen (no title bar)
    screen = pygame.display.set_mode(
        (PROJECTOR_W, PROJECTOR_H),
        pygame.FULLSCREEN | pygame.NOFRAME
    )
    pygame.display.set_caption("Vein Projector")
    clock = pygame.time.Clock()

    blank = pygame.Surface((PROJECTOR_W, PROJECTOR_H))
    blank.fill((0, 0, 0))
    screen.blit(blank, (0, 0))
    pygame.display.flip()

    last_jpeg = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

        with lock:
            cur_jpeg = state['processed_jpeg']

        if cur_jpeg and cur_jpeg is not last_jpeg:
            last_jpeg = cur_jpeg
            try:
                surf = build_projection_surface(cur_jpeg)
                if surf:
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
            except Exception as e:
                print(f"Projection error: {e}")

        clock.tick(30)   # 30 fps display update cap

# ── FLASK ROUTES (control panel accessible from phone / laptop) ───────────────
CONTROL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Vein Viewer Control</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:sans-serif;background:#111;color:#eee;padding:16px}
    h2{color:#4fc3f7}
    .row{display:flex;gap:8px;align-items:center;margin:8px 0}
    label{width:160px;display:inline-block}
    input[type=range]{width:200px}
    button{background:#4fc3f7;border:none;padding:8px 20px;cursor:pointer;border-radius:4px}
    img{border:1px solid #333;margin:4px}
  </style>
</head>
<body>
  <h2>🩸 Vein Viewer Control</h2>
  <div class="row"><label>Exposure</label>
    <input type="range" id="exposure" min="500" max="20000" step="100" value="3500"
      oninput="send('exposure',+this.value)"> <span id="vexposure">3500</span>
  </div>
  <div class="row"><label>Gain</label>
    <input type="range" id="gain" min="1" max="8" step="0.1" value="1.5"
      oninput="send('gain',+this.value)"> <span id="vgain">1.5</span>
  </div>
  <div class="row"><label>Focus (lens pos)</label>
    <input type="range" id="lens_position" min="2" max="14" step="0.5" value="8"
      oninput="send('lens_position',+this.value)"> <span id="vlens_position">8</span>
  </div>
  <div class="row"><label>Proj Brightness</label>
    <input type="range" id="brightness" min="0.3" max="3.0" step="0.1" value="1.0"
      oninput="send('brightness',+this.value)"> <span id="vbrightness">1.0</span>
  </div>
  <div class="row"><label>Proj Offset X</label>
    <input type="range" id="proj_offset_x" min="-400" max="400" step="5" value="0"
      oninput="send('proj_offset_x',+this.value)"> <span id="vproj_offset_x">0</span>
  </div>
  <div class="row"><label>Proj Offset Y</label>
    <input type="range" id="proj_offset_y" min="-300" max="300" step="5" value="0"
      oninput="send('proj_offset_y',+this.value)"> <span id="vproj_offset_y">0</span>
  </div>
  <div class="row"><label>Proj Scale</label>
    <input type="range" id="proj_scale" min="0.5" max="2.0" step="0.05" value="1.0"
      oninput="send('proj_scale',+this.value)"> <span id="vproj_scale">1.0</span>
  </div>
  <div class="row"><label>Proj Rotate</label>
    <select id="proj_rotate" onchange="send('proj_rotate',+this.value)">
      <option value="0">0°</option><option value="90">90°</option>
      <option value="180">180°</option><option value="270">270°</option>
    </select>
  </div>
  <br>
  <img src="/live_feed" width="45%" title="Live IR">
  <img src="/processed_feed" width="45%" title="Processed">

  <script>
    function send(key, val){
      document.getElementById('v'+key).textContent = val;
      fetch('/settings',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({[key]:val})});
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(CONTROL_HTML)

@app.route('/live_feed')
def live_feed():
    def gen():
        while True:
            with lock:
                j = state['live_jpeg']
            if j:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + j + b'\r\n'
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_feed')
def processed_feed():
    def gen():
        while True:
            with lock:
                j = state['processed_jpeg']
            if j:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + j + b'\r\n'
            time.sleep(0.1)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['POST'])
def update_settings():
    data = request.json
    for k, v in data.items():
        if k in settings:
            settings[k] = v
    _apply_camera_controls()
    return jsonify({"status": "ok"})

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_camera()
    threading.Thread(target=capture_loop,     daemon=True).start()
    threading.Thread(target=cloud_sync_worker, daemon=True).start()
    # Flask runs in its own thread so projector_loop owns the main thread
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=8080, threaded=True),
        daemon=True
    ).start()
    # Projector loop MUST be on main thread (pygame requirement)
    projector_loop()