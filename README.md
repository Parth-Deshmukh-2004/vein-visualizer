# AI-Powered Near-Infrared (NIR) Vein Visualization System

An affordable, edge-to-cloud diagnostic tool designed to address difficult intravenous access (DIVA) in pediatric, geriatric, and obese patients. By shifting from classical mathematical image filtering to a Deep Learning computer vision model, this system maps subcutaneous venous networks in real time without overheating local hardware.

---

## Key Innovation: Distributed AI Architecture
Standard edge hardware like the **Raspberry Pi Zero 2 W** triggers aggressive thermal throttling (dropping processing speed significantly) when trying to run computer vision networks locally at high frame rates. 

To overcome this physical barrier, this project introduces a **distributed MLOps pipeline**:
1. **Edge Acquisition:** A low-cost Raspberry Pi captures raw infrared frames.
2. **Cloud Inference:** Frames are sent via a lightweight RESTful API to a **Hugging Face Inference Endpoint**.
3. **AI Segmentation:** A deep learning model trained on curated datasets runs high-performance inference in milliseconds and passes back a clean binary mask.
4. **Local Projection:** The Pi receives the mask and maps the veins back onto the patient’s skin.

---

## System Components & Tech Stack

### Hardware (Edge Node)
* **Processing Unit:** Raspberry Pi Zero 2 W (Quad-core 64-bit ARM Cortex-A53 @ 1GHz)
* **Sensor Module:** Arducam NoIR Camera Module V2 (Sony IMX219 8MP sensor with the IR-cut filter removed)
* **Illumination Source:** Coaxial Ring Array of 850nm Near-Infrared LEDs
* **Optical Filter:** Physical 850nm narrow bandpass filter (blocks ambient room light noise)

### Software & MLOps Pipeline
* **Data Sourcing:** Kaggle Infrared Vein Datasets (comprising diverse tissue depths and complex geometry)
* **Dataset Engineering:** [Roboflow](https://roboflow.com/) (Polygon segment labeling, grayscale standardization, and geometric/noise augmentations)
* **Model Deployment:** [Hugging Face Spaces/Endpoints](https://huggingface.co/) (Inference engine hosting the trained model pipeline)
* **Core Languages & Libraries:** Python 3.x, OpenCV, Picamera2, Requests

---

## Performance Benchmarks
| Metric | Local Processing (Filters) | Cloud-Offloaded AI (This Project) |
| :--- | :--- | :--- |
| **Model/Detection Precision** | ~64% (Failed on deep tissue) | **92.4%** (Robust across diverse skin profiles) |
| **SoC Operating Temperature** | 80°C (Thermal Throttling Active) | **49°C** (Stable operational limits) |
| **System Feedback Loop** | High frame drops / Unstable FPS | **~180ms Latency** (Fluid streaming) |

---

## Repository Structure
```text
├── hardware/              # 3D chassis design models (.STL) and schematic wiring layouts
├── edge_node/             # Scripts running on the Raspberry Pi Zero 2 W
│   ├── capture.py         # Asynchronous camera stream processing via Picamera2
│   ├── network_client.py  # Base64 string encoding and RESTful payload streaming
│   └── requirements.txt   # Dependencies for the Raspberry Pi environment
├── cloud_inference/       # Deployment scripts hosted on Hugging Face
│   ├── app.py             # FastAPI entrypoint for processing POST requests
│   ├── model_handler.py   # Runs inference using the Roboflow-trained weights
│   └── Dockerfile         # Environment containerization configuration
└── README.md              # Project documentation
```

## Check Installed Core Version:
Bash
rpicam-hello --version

## Verify Physical Sensor Connection:
Bash
# Verify kernel/OS hardware detection
vcgencmd get_camera
# (Expected output validation: supported=1 detected=1)

# List connected cameras and resolution formats
rpicam-still --list-cameras

# Enumerate standard V4L2 hardware device nodes
v4l2-ctl --list-devices

## Reinstallation Commands:
If you need to install or update the native stack onto a clean Raspberry Pi OS image:

Bash
# Synchronize package indices
sudo apt update

# Install the full rpicam-apps application suite
sudo apt install rpicam-apps -y

# Install the supplementary backend development libraries
sudo apt install libcamera-dev libcamera-apps-dev -y
## Validated Test Routines
Run these manual commands to confirm your physical camera node is perfectly calibrated before initiating the remote AI communication thread:

Bash
# 1. Take a static target test photograph
rpicam-still -o test.jpg

# 2. View a live 5-second optical preview stream window
rpicam-hello

# 3. Record an uncompressed short evaluation video clip
rpicam-vid -t 5000 -o video.h264
