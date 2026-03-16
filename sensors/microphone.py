import os
import csv
import time
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from ai_edge_litert.interpreter import Interpreter

class Microphone:
    """
    Microphone class using webcam microphone + YAMNet TFLite
    Returns confidence scores for weighted fusion.
    """

    def __init__(
        self,
        yamnet_model_path="models/yamnet.tflite",
        class_map_path="models/yamnet_class_map.csv",
        sample_rate=16000,
        chunk_duration=0.975,
        detection_threshold=0.3,
        event_hold_time=2.0,
        loudness_threshold=0.02,
        impact_threshold=0.35,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.detection_threshold = detection_threshold
        self.event_hold_time = event_hold_time
        self.loudness_threshold = loudness_threshold
        self.impact_threshold = impact_threshold

        self.audio_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.stream = None

        # Detection / fusion state
        self.latest_confidence = 0.0
        self.held_confidence = 0.0
        self.latest_label = "none"
        self.last_detection_time = 0.0
        self.last_result = {
            "sensor": "microphone",
            "confidence": 0.0,
            "instant_confidence": 0.0,
            "label": "none",
            "detected": False,
            "timestamp": time.time(),
            "rms": 0.0,
            "peak": 0.0,
        }
        
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

        # Load TFLite model
        self.interpreter = Interpreter(model_path=self.yamnet_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class names
        self.class_names = self._load_class_map(self.class_map_path)

        # Emergency-related keywords
        self.target_keywords = [
            "shout",
            "yell",
            "scream",
            "cry",
            "bang",
            "thump",
            "thud",
            "crash",
            "slap",
            "impact",
        ]

        # ---------------------------------------------------
        # Initialize bandpass filter ONCE (optimization)
        # This avoids recomputing filter every audio chunk
        # ---------------------------------------------------
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
        """
        Precompute bandpass filter for impact-like sounds.
        """
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

        audio = indata[:, 0]  # mono
        self.audio_queue.put(audio.copy())

    def start(self, device=None):
        """
        Start microphone stream and processing thread.
        """
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
        """
        Stop microphone stream and processing thread.
        """
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[WARN] Error stopping stream: {e}")
            finally:
                self.stream = None

        print("[INFO] Microphone stopped.")

    def _process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)

                instant_conf, label, rms, peak = self.detect_fall_sound(audio_chunk)
                self.latest_confidence = instant_conf
                self.latest_label = label

                if instant_conf >= self.detection_threshold:
                    self.held_confidence = instant_conf
                    self.last_detection_time = time.time()
                else:
                    if time.time() - self.last_detection_time > self.event_hold_time:
                        self.held_confidence = 0.0

                detected = self.held_confidence >= self.detection_threshold

                self.last_result = {
                    "sensor": "microphone",
                    "confidence": float(self.held_confidence),
                    "instant_confidence": float(self.latest_confidence),
                    "label": self.latest_label,
                    "detected": detected,
                    "timestamp": time.time(),
                    "rms": float(rms),
                    "peak": float(peak),
                }

                if instant_conf >= self.detection_threshold:
                    print(
                        f"[AUDIO] label={label} | instant={instant_conf:.3f} "
                        f"| held={self.held_confidence:.3f} | RMS={rms:.3f} | Peak={peak:.3f}"
                    )

            except queue.Empty:
                if time.time() - self.last_detection_time > self.event_hold_time:
                    self.held_confidence = 0.0
                    self.last_result["confidence"] = 0.0
                    self.last_result["detected"] = False

            except Exception as e:
                print(f"[ERROR] Audio processing failed: {e}")
                

    def detect_fall_sound(self, audio):
        """
        Returns:
            instant_confidence, best_label, rms, peak
        """

        REQUIRED_SAMPLES = 15600
        audio = np.asarray(audio, dtype=np.float32)

        # Pad or trim audio to match model input size      
        if len(audio) < REQUIRED_SAMPLES:
            audio = np.pad(audio, (0, REQUIRED_SAMPLES - len(audio)), mode="constant")
        elif len(audio) > REQUIRED_SAMPLES:
            audio = audio[:REQUIRED_SAMPLES]
            
        # ---------------------------------------------------
        # Step 1: Band-pass filter
        # Removes low rumble (fans, AC) and high-frequency noise
        # Keeps frequencies where impacts usually occur
        # ---------------------------------------------------

        # Band-pass filter
        audio = self.bandpass_filter(audio).astype(np.float32)

        # ---------------------------------------------------
        # Step 2: Loudness gate (ignore quiet background noise)
        # ---------------------------------------------------
        # Signal features
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        # ---------------------------------------------------
        # Step 3: Detect sudden impact peaks
        # Falls usually produce a spike in amplitude
        # ---------------------------------------------------
        if rms < 0.005:
            return 0.0, "too_quiet", rms, peak

        # TFLite inference
        input_index = self.input_details[0]["index"]
        expected_shape = self.input_details[0]["shape"]
        
        # ---------------------------------------------------
        # Step 4: Run YAMNet inference
        # ---------------------------------------------------

        if len(expected_shape) == 1:
            input_tensor = audio
        else:
            input_tensor = np.expand_dims(audio, axis=0)

        self.interpreter.set_tensor(input_index, input_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        if output_data.ndim == 1:
            scores = output_data
        else:
            scores = output_data[0]

        # Best target class among top predictions
        best_keyword_score = 0.0
        best_label = "none"

        top_indices = np.argsort(scores)[::-1][:10]

        # ---------------------------------------------------
        # Step 5: Check if predicted sound matches emergency types
        # ---------------------------------------------------
        for idx in top_indices:
            score = float(scores[idx])
            label = self.class_names[idx].lower()

            if any(keyword in label for keyword in self.target_keywords):
                if score > best_keyword_score:
                    best_keyword_score = score
                    best_label = self.class_names[idx]

        # Normalize signal features to 0..1
        loudness_score = min(rms / self.loudness_threshold, 1.0) if self.loudness_threshold > 0 else 0.0
        impact_score = min(peak / self.impact_threshold, 1.0) if self.impact_threshold > 0 else 0.0

        # Final confidence
        instant_confidence = (
            0.6 * best_keyword_score +
            0.2 * loudness_score +
            0.2 * impact_score
        )

        instant_confidence = float(np.clip(instant_confidence, 0.0, 1.0))

        return instant_confidence, best_label, rms, peak

    def get_confidence(self):
        return float(self.latest_confidence)

    def get_held_confidence(self):
        return float(self.held_confidence)

    def is_emergency_detected(self):
        return self.held_confidence >= self.detection_threshold

    def get_detection_result(self):
        """
        Standard output format for weighted fusion.
        """
        return {
            "sensor": self.last_result["sensor"],
            "confidence": float(self.last_result["confidence"]),
            "instant_confidence": float(self.last_result["instant_confidence"]),
            "label": self.last_result["label"],
            "detected": bool(self.last_result["detected"]),
            "timestamp": float(self.last_result["timestamp"]),
            "rms": float(self.last_result["rms"]),
            "peak": float(self.last_result["peak"]),
        }


if __name__ == "__main__":
    mic = None

    try:
        mic = Microphone()

        print("[INFO] Starting microphone test...")
        print("[INFO] Make a loud impact-like sound or shout to test detection.")
        print("[INFO] Press Ctrl+C to stop.\n")

        mic.start()

        while True:
            result = mic.get_detection_result()
            print(result)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        if mic is not None:
            mic.stop()
