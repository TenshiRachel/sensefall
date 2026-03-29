from flask import Flask, Response, jsonify
import cv2
import json
import os
import time

app = Flask(__name__)

FRAME_PATH = "/tmp/latest_frame.jpg"
STATUS_PATH = "/tmp/camera_status.json"

def gen_frames():
    while True:
        if not os.path.exists(FRAME_PATH):
            time.sleep(0.1)
            continue

        with open(FRAME_PATH, "rb") as f:
            frame_bytes = f.read()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.1)  # ~10fps polling

@app.route('/')
def home():
    return '<h2>Pi Camera</h2><img src="/video_feed" style="max-width:100%;">'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    if not os.path.exists(FRAME_PATH):
        return ("No frame available", 503)
    with open(FRAME_PATH, "rb") as f:
        frame_bytes = f.read()

    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return Response(frame_bytes, mimetype='image/jpeg')

    capture_label = 'Captured: unavailable'
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, 'r') as status_file:
                status_data = json.load(status_file)
            ts = status_data.get('timestamp')
            if ts is not None:
                capture_label = f"Captured: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
        except Exception:
            pass

    cv2.putText(
        frame,
        capture_label,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        return Response(frame_bytes, mimetype='image/jpeg')
    return Response(encoded.tobytes(), mimetype='image/jpeg')

@app.route('/status')
def status():
    if not os.path.exists(STATUS_PATH):
        return jsonify({"error": "No status available"}), 503
    with open(STATUS_PATH, "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
