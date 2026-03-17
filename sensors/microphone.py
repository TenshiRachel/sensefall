import os
import csv
import time
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from tflite_runtime.interpreter import Interpreter


class Microphone:
    """
    Microphone class using webcam microphone + YAMNet TFLite.
    Shows debug predictions and simple fall-risk / normal-sound classification.
    """

    def __init__(
        self,
        yamnet_model_path="models/yamnet.tflite",
        class_map_path="models/yamnet_class_map.csv",
        sample_rate=16000,
        chunk_duration=0.975,
        detection_threshold=0.20,
        event_hold_time=2.0,
        min_alert_peak=0.08,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.detection_threshold = detection_threshold
        self.event_hold_time = event_hold_time
        self.min_alert_peak = min_alert_peak

        self.audio_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.stream = None

        self.emergency_detected = False
        self.last_detection_time = 0.0

        # Resolve model paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)

        self.yamnet_model_path = (
            yamnet_model_path
            if os.path.isabs(yamnet_model_path)
            else os.path.join(project_root, yamnet_model_path)
        )

        self.class_map_path = (
            class_map_path
            if os.path.isabs(class_map_path)
            else os.path.join(project_root, class_map_path)
        )

        if not os.path.exists(self.yamnet_model_path):
            raise FileNotFoundError(f"YAMNet model not found: {self.yamnet_model_path}")

        if not os.path.exists(self.class_map_path):
            raise FileNotFoundError(f"Class map not found: {self.class_map_path}")

        # Load interpreter
        self.interpreter = Interpreter(model_path=self.yamnet_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load labels
        self.class_names = self._load_class_map(self.class_map_path)

        # Keywords for alert-like sounds
        self.fall_keywords = [
            "bang",
            "thump",
            "thud",
            "crash",
            "slap",
            "smack",
            "breaking",
            "glass",
            "explosion",
            "fireworks",
            "yell",
            "shout",
            "scream",
            "cry",
        ]

        # Keywords for normal household / background sounds
        self.normal_keywords = [
            "speech",
            "television",
            "tv",
            "clapping",
            "hands",
            "tap",
            "door",
            "music",
            "fan",
            "inside",
            "room",
            "conversation",
            "male speech",
            "female speech",
        ]
        
        self._init_bandpass_filter()

    def _load_class_map(self, path):
        class_names = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                class_names.append(row[2])
        return class_names

    def _init_bandpass_filter(self):
        lowcut = 100
        highcut = 2000
        order = 4

        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        self.b, self.a = butter(order, [low, high], btype="band")

    def bandpass_filter(self, audio):
        return lfilter(self.b, self.a, audio)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO STATUS] {status}")

        audio = indata[:, 0]

        try:
            self.audio_queue.put_nowait(audio.copy())
        except queue.Full:
            pass

    def start(self, device=None):
        if self.running:
            print("[INFO] Microphone already running.")
            return

        self.running = True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
                blocksize=self.chunk_samples,
                device=device,
            )
            self.stream.start()
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start audio stream: {e}")

        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

        print("[INFO] Microphone started.")

    def stop(self):
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[WARN] Error stopping stream: {e}")
            finally:
                self.stream = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1)

        print("[INFO] Microphone stopped.")

    def _process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                detected, status_label, rms, peak, top_predictions = self.detect_sound(audio_chunk)

                if detected:
                    self.emergency_detected = True
                    self.last_detection_time = time.time()
                else:
                    if time.time() - self.last_detection_time > self.event_hold_time:
                        self.emergency_detected = False

                timestamp = time.strftime("%H:%M:%S")

                print(f"\n[MIC] {timestamp}")
                print(f"RMS: {rms:.4f} | Peak: {peak:.4f} | Emergency: {self.emergency_detected}")
                print("Top predictions:")

                for label, score in top_predictions:
                    print(f"  - {label}: {score * 100:.2f}%")

                if detected:
                    print(f"[ALERT] FALL RISK SOUND DETECTED -> {status_label}")
                else:
                    if status_label.startswith("normal:"):
                        print(f"[INFO] Likely household/background sound -> {status_label}")
                    elif status_label == "too_quiet":
                        print("[INFO] Too quiet / no meaningful sound")
                    else:
                        print("[INFO] No fall-risk sound detected")

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[ERROR] Audio processing failed: {e}")
                

    def detect_sound(self, audio):
        """
        Returns:
            detected, status_label, rms, peak, top_predictions
        """
        REQUIRED_SAMPLES = 15600
        audio = np.asarray(audio, dtype=np.float32)

        # Pad or trim
        if len(audio) < REQUIRED_SAMPLES:
            audio = np.pad(audio, (0, REQUIRED_SAMPLES - len(audio)), mode="constant")
        elif len(audio) > REQUIRED_SAMPLES:
            audio = audio[:REQUIRED_SAMPLES]

        # Band-pass filter
        audio = self.bandpass_filter(audio).astype(np.float32)

        # Features
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        if rms < 0.003:
            return False, "too_quiet", rms, peak, []

        # Prepare input
        input_shape = self.input_details[0]["shape"]

        if len(input_shape) == 1:
            input_tensor = audio.astype(np.float32)
        else:
            input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        if output_data.ndim == 1:
            scores = output_data
        else:
            scores = output_data[0]

        # Top 5 predictions
        top_indices = np.argsort(scores)[::-1][:5]
        top_predictions = []

        fall_score = 0.0
        normal_score = 0.0
        best_fall_label = "none"
        best_normal_label = "none"

        for idx in top_indices:
            score = float(scores[idx])
            label = self.class_names[idx]
            label_lower = label.lower()
            top_predictions.append((label, score))

            if any(keyword in label_lower for keyword in self.fall_keywords):
                if score > fall_score:
                    fall_score = score
                    best_fall_label = label

            if any(keyword in label_lower for keyword in self.normal_keywords):
                if score > normal_score:
                    normal_score = score
                    best_normal_label = label

        detected = False
        status_label = "uncertain"

        # Alert condition
        if fall_score >= self.detection_threshold and peak >= self.min_alert_peak:
            detected = True
            status_label = f"fall_risk: {best_fall_label}"

        # Normal sound condition
        elif normal_score >= 0.10:
            status_label = f"normal: {best_normal_label}"

        else:
            status_label = "uncertain"

        return detected, status_label, rms, peak, top_predictions

    def is_emergency_detected(self):
        return self.emergency_detected


if __name__ == "__main__":
    mic = None

    try:
        mic = Microphone()

        print("[INFO] Starting microphone test...")
        print("[INFO] Speak, clap, shout, or make impact sounds to test.")
        print("[INFO] Press Ctrl+C to stop.")

        mic.start()

        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        if mic is not None:
            mic.stop()
