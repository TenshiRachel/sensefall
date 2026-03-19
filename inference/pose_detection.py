from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import time


class PoseEstimator:
    def __init__(self, model_path='models/movenet_model.tflite'):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.prev_center = None
        self.smoothed_center_y = None
        self.prev_time = time.time()

    # All 17 MoveNet keypoints for expanded obstruction handling
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1, "right_eye": 2,
        "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    }

    # Keypoint groups in priority order for body center estimation
    # If higher priority group isn't visible, fall back to next
    CENTER_KEYPOINT_GROUPS = [
        ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],  # ideal
        ["left_shoulder", "right_shoulder"],                            # upper body only
        ["left_hip", "right_hip"],                                      # lower body only
        ["left_shoulder", "left_hip"],                                  # left side only
        ["right_shoulder", "right_hip"],                                # right side only
        ["left_shoulder", "right_hip"],                                 # diagonal
        ["right_shoulder", "left_hip"],                                 # diagonal
        ["left_knee", "right_knee"],                                    # legs only
    ]

    # Keypoint pairs used for body orientation ratio
    RATIO_PAIRS = [
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),      # ideal
        ("left_shoulder", "right_shoulder", "left_knee", "right_knee"),    # no hips
        ("left_hip", "right_hip", "left_knee", "right_knee"),              # no shoulders
    ]

    def estimate_pose(self, frame):
        """
        Run the MoveNet pose estimation on a frame
        """
        img = cv2.resize(frame, (192, 192))
        img = np.expand_dims(img.astype(np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
        return keypoints

    def _get_visible_keypoints(self, keypoints, names, conf_threshold):
        """Return dict of visible keypoints by name."""
        visible = {}
        for name in names:
            idx = self.KEYPOINT_INDICES[name]
            kp = keypoints[idx]
            if kp[2] >= conf_threshold:
                visible[name] = kp
        return visible

    def _estimate_body_center(self, keypoints, conf_threshold, frame_width, frame_height):
        """
        Estimate body center using best available keypoint group.
        Returns (center_x, center_y, group_used) or None if insufficient keypoints.
        """
        for group in self.CENTER_KEYPOINT_GROUPS:
            visible = self._get_visible_keypoints(keypoints, group, conf_threshold)
            if len(visible) >= 2:
                xs = [kp[1] * frame_width for kp in visible.values()]
                ys = [kp[0] * frame_height for kp in visible.values()]
                return np.mean(xs), np.mean(ys), group
        return None

    def _estimate_body_ratio(self, keypoints, conf_threshold):
        """
        Estimate shoulder_width / body_height ratio using best available pairs.
        Returns ratio or None if insufficient keypoints.
        """
        for top1, top2, bot1, bot2 in self.RATIO_PAIRS:
            visible = self._get_visible_keypoints(
                keypoints, [top1, top2, bot1, bot2], conf_threshold
            )
            # Need at least one from top and one from bottom
            top_visible = [k for k in [top1, top2] if k in visible]
            bot_visible = [k for k in [bot1, bot2] if k in visible]

            if not top_visible or not bot_visible:
                continue

            top_y = np.mean([visible[k][0] for k in top_visible])
            bot_y = np.mean([visible[k][0] for k in bot_visible])
            top_x = np.mean([visible[k][1] for k in top_visible])
            bot_x = np.mean([visible[k][1] for k in bot_visible])

            body_height = abs(top_y - bot_y)
            body_width = abs(top_x - bot_x) if len(top_visible) > 1 else abs(
                visible[top_visible[0]][1] - visible[bot_visible[0]][1]
            )

            if body_height == 0:
                continue

            return body_width / body_height

        return None

    def detect_fall_pose(self, keypoints, frame_width, frame_height):
        """
        Detect fall using:
        1. Body orientation ratio (lying vs standing)
        2. Downward velocity of body center
        Works with partial keypoint visibility - minimum 2 keypoints needed.
        """
        CONF_THRESHOLD = 0.2
        FALL_RATIO_THRESHOLD = 0.9
        FALL_VELOCITY_THRESHOLD = 20
        SMOOTHING = 0.7

        keypoints = keypoints[0][0]

        # --- Body center estimation ---
        result = self._estimate_body_center(
            keypoints, CONF_THRESHOLD, frame_width, frame_height
        )
        if result is None:
            return 0.0

        center_x, center_y, group_used = result

        # --- Body ratio estimation ---
        body_ratio = self._estimate_body_ratio(keypoints, CONF_THRESHOLD)
        if body_ratio is None:
            return 0.0

        # --- Smooth center_y ---
        if self.smoothed_center_y is None:
            self.smoothed_center_y = center_y
        else:
            self.smoothed_center_y = (
                SMOOTHING * self.smoothed_center_y +
                (1 - SMOOTHING) * center_y
            )

        # --- Velocity ---
        velocity = 0
        current_time = time.time()
        if self.prev_center is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                velocity = (self.smoothed_center_y - self.prev_center[1]) / dt

        self.prev_center = (center_x, self.smoothed_center_y)
        self.prev_time = current_time
        
        # print(f"Velocity: {velocity}, Body ratio: {body_ratio}")
        
        # --- Fall decision ---
        if body_ratio > FALL_RATIO_THRESHOLD and velocity > FALL_VELOCITY_THRESHOLD:
            print(f"[POSE] POSSIBLE FALL DETECTED (Keypoints used: {group_used})")
            return 1.0

        return 0.0
