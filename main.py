import time
from threading import Thread, Lock
from collections import deque

import requests
import os
from dotenv import load_dotenv
from datetime import datetime

from sensors.camera import Camera
from sensors.mmWave import HumanTrackerWithVelocity
from sensors.microphone import Microphone

from inference.pose_detection import PoseEstimator
from inference.weighted_fusion import WeightedFusion

from services.alert_service import send_fall_alert
load_dotenv()


# ========================
# INIT
# ========================
camera = Camera()
radar = HumanTrackerWithVelocity(bus=1, addr=0x52, busy_pin=4)
mic = Microphone()

pose_model = PoseEstimator()
fusion = WeightedFusion()

FUSION_FALL_THRESHOLD = 0.7
FALL_COOLDOWN = 5 # seconds

last_fall_time = 0

# Sliding windows (temporal alignment)
camera_buffer = deque(maxlen=5)
radar_buffer = deque(maxlen=5)
mic_buffer = deque(maxlen=5)

lock = Lock()

running = True

# ========================
# CAMERA THREAD
# ========================
def camera_worker():
    while running:
        t_capture = time.perf_counter()
        frame = camera.get_frame()
        if frame is None:
            # print("Frame error")
            continue

        person_visible = camera.detect_person(frame)

        if not person_visible:
            # print("No person detected")
            conf = None
            t_capture = None
        else:
            h, w, _ = frame.shape
            keypoints = pose_model.estimate_pose(frame)
            conf = pose_model.detect_fall_pose(keypoints, w, h)

        with lock:
            camera_buffer.append((conf, t_capture))

        time.sleep(0.2)  # ~5 FPS


# ========================
# RADAR THREAD
# ========================
def radar_worker():
    radar.init()

    while running:
        t_capture = time.perf_counter()
        tracked = radar.track_humans_with_velocity()
        
        if len(tracked) == 0:
            conf = None
            t_capture = None
        else:
            falls = radar.detect_fall(tracked)
            conf = falls[0]['fall_confidence'] if len(falls) > 0 else None

        with lock:
            radar_buffer.append((conf, t_capture))
            
        time.sleep(0.1)  # faster than camera


# ========================
# MICROPHONE THREAD
# ========================
def mic_worker():
    mic.start()

    while running:
        t_capture = time.perf_counter()
        conf = mic.get_confidence()

        with lock:
            mic_buffer.append((conf, t_capture))

        time.sleep(0.1)


# ========================
# FUSION THREAD
# ========================
def fusion_worker():
    global last_fall_time
    while running:
        t_fusion_start = time.perf_counter()
        with lock:
            cam_entries = [(v, t) for v, t in camera_buffer if v is not None]
            rad_entries = [(v, t) for v, t in radar_buffer if v is not None]
            mic_entries = [(v, t) for v, t in mic_buffer if v is not None]

        # Extract scores and find earliest capture timestamp
        cam_vals = [v for v, _ in cam_entries]
        rad_vals = [v for v, _ in rad_entries]
        mic_vals = [v for v, _ in mic_entries]

        # Earliest capture time across all sensors that contributed
        all_timestamps = (
            [t for _, t in cam_entries] +
            [t for _, t in rad_entries] +
            [t for _, t in mic_entries]
        )
        t_first_capture = min(all_timestamps) if all_timestamps else None

        # Temporal smoothing
        camera_conf = max(cam_vals) if cam_vals else None
        radar_conf = max(rad_vals) if rad_vals else None
        mic_conf = max(mic_vals) if mic_vals else None

        # Early exit
        if camera_conf is None and radar_conf is None and (mic_conf is None or mic_conf < 0.2):
            time.sleep(0.1)
            continue

        final_score = fusion.fuse(
            camera_conf=camera_conf,
            mmwave_conf=radar_conf,
            mic_conf=mic_conf
        )
        t_fusion_end = time.perf_counter()
        current_time = time.time()
        print(f"Camera: {camera_conf}, Mic: {mic_conf}, Mmwave:{radar_conf}")
        # print(f"Final score: {final_score}")

        if final_score > FUSION_FALL_THRESHOLD:
            if current_time - last_fall_time > FALL_COOLDOWN:
                event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print("[FUSION] FALL DETECTED")
                t_email_start = time.perf_counter()

                # Send email
                send_fall_alert(
                    confidence=final_score,
                    trigger_source="fusion",
                    event_time=event_time,
                    sensor="fusion",
                    sensor_scores={
                        "camera": camera_conf,
                        "mic": mic_conf
                    },
                    device_id=os.getenv("CLOUD_DEVICE_ID", "home_pi_01"),
                    location="home",
                    extra_notes=f"Camera={camera_conf}, Radar={radar_conf}, Mic={mic_conf}"
                )
		
                t_email_end = time.perf_counter()
                t_dashboard_start = time.perf_counter()

                # Sync FALL event
                sent_event_to_dashboard(
                    event_time=event_time,
                    event_type="fall",
                    confidence=final_score,
                    metadata={
                        "trigger": "fusion",
                        "sensor": "fusion",
                        "sensor_scores": {
                            "camera": camera_conf,
                            "radar": radar_conf,
                            "mic": mic_conf
                        }
                    }
                )

                # Sync EMAIL event
                sent_event_to_dashboard(
                    event_time=event_time,
                    event_type="email",
                    confidence=None,
                    metadata={
                        "trigger": "fusion",
                        "sensor": "fusion",
                        "status": "sent",
                        "sensor_scores": {
                            "camera": camera_conf,
                            "radar": radar_conf,
                            "mic": mic_conf
                        }
                    }
                )

                t_dashboard_end = time.perf_counter()
                last_fall_time = current_time

                # ========================
                # TIMING REPORT
                # ========================
                if t_first_capture:
                    t_capture_to_fusion = (t_fusion_start - t_first_capture) * 1000
                    t_fusion_duration = (t_fusion_end - t_fusion_start) * 1000
                    t_dashboard_duration = (t_dashboard_end - t_dashboard_start) * 1000
                    t_email_duration = (t_email_end - t_email_start) * 1000
                    t_total = (t_dashboard_end - t_first_capture) * 1000

                    print(
                        f"\n[TIMING] ---- Fall Event Latency ----\n"
                        f"  Capture to Fusion start : {t_capture_to_fusion:.2f} ms\n"
                        f"  Fusion processing      : {t_fusion_duration:.2f} ms\n"
                        f"  Delivery (dashboard)   : {t_dashboard_duration:.2f} ms\n"
                        f"  Delivery (email)         : {t_email_duration:.2f} ms\n"
                        f"  Total (capture to sent) : {t_total:.2f} ms\n"
                        f"--------------------------------------"
                    )
            else:
                # Optional debug
                print("Cooldown active, skipping alert")

        time.sleep(0.1)
# ========================
# SMTP ALERT
# ========================
def sent_event_to_dashboard(event_time, event_type, confidence=None, metadata=None):
    DASHBOARD_URL = os.getenv("CLOUD_SYNC_URL", os.getenv("DASHBOARD_URL", "http://localhost:5000"))
    api_key = os.getenv("CLOUD_SYNC_API_KEY", "")
    url = f"{DASHBOARD_URL.rstrip('/')}/api/event"

    payload = {
        "event_time": event_time,
        "event_type": event_type,
        "confidence": confidence,
        "device_id": os.getenv("CLOUD_DEVICE_ID", "home_pi_01"),
        "source": "edge",
        "metadata": metadata or {}
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = requests.post(url, json=payload,headers=headers, timeout=int(os.getenv("CLOUD_SYNC_TIMEOUT_SEC", "5")))
        print("[INFO] Sent to dashboard")
    except Exception as e:
        print(f"[ERROR] Dashboard sent failed:", e)


# ========================
# START THREADS
# ========================
threads = [
    Thread(target=camera_worker),
    Thread(target=radar_worker),
    Thread(target=mic_worker),
    Thread(target=fusion_worker),
]

for t in threads:
    t.start()

# ========================
# CLEAN EXIT
# ========================
try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    running = False

    for t in threads:
        t.join()

    camera.release()
    mic.stop()
