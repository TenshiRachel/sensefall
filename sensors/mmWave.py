import smbus
import time
import sys
from collections import deque

from sensors.pythonlibrary.A121_Distance_Detector import *

# ===============================
#          parameters
# ===============================

HUMAN_DISTANCE_MIN_MM = 0
HUMAN_DISTANCE_MAX_MM = 3000

HUMAN_STRENGTH_MIN = 0
HUMAN_STRENGTH_MAX = 10

# SENSOR_TO_FLOOR_MM = 1000 #for manual testing
FALL_HEIGHT_THRESHOLD = 400

FALL_VELOCITY_THRESHOLD = 800
HUMAN_MOVEMENT_TOLERANCE = 50

FALL_CONFIDENCE_THRESHOLD = 0.7


class HumanDetector(A121_Distance_Detector):

    def __init__(self, bus=1, addr=0x52, busy_pin=4):
        super().__init__(bus, addr, busy_pin)

    def detect_humans(self):

        self.set_command(SensorCommand.MEASURE_DISTANCE)

        while True:
            status = self.get_detector_status()
            if (status & ALL_ERROR) == 0:
                break
            time.sleep(0.01)

        result = self.get_distance_result()
        num_peaks = result & SensorDistanceResult.NUM_DISTANCES

        humans = []

        if (result & SensorDistanceResult.MEASURE_DISTANCE_ERROR) == 0 and num_peaks > 0:

            for i in range(num_peaks):

                distance = self.read_u32(REG_PEAK0_DISTANCE + i)
                strength_u32 = self.read_u32(REG_PEAK0_STRENGTH + i)

                strength = strength_u32 - 0x100000000 if strength_u32 & 0x80000000 else strength_u32

                if HUMAN_DISTANCE_MIN_MM <= distance <= HUMAN_DISTANCE_MAX_MM:

                    humans.append({
                        "distance_mm": int(distance),
                        "strength": strength / 1000.0
                    })

        return humans
        
class HumanTracker(HumanDetector):

    def __init__(self, bus=1, addr=0x52, busy_pin=4):
        super().__init__(bus, addr, busy_pin)
        self.previous_humans = []

    def track_humans(self):

        humans = self.detect_humans()
        tracked = []

        for h in humans:

            closest_prev = None
            min_dist = 10000

            for p in self.previous_humans:
                d_diff = abs(p["distance_mm"] - h["distance_mm"])

                if d_diff < min_dist:
                    min_dist = d_diff
                    closest_prev = p

            if closest_prev and min_dist < 200:

                tracked.append({
                    "distance_mm": h["distance_mm"],
                    "strength": h["strength"],
                    "id": closest_prev["id"]
                })

            else:

                tracked.append({
                    "distance_mm": h["distance_mm"],
                    "strength": h["strength"],
                    "id": id(h)
                })

        self.previous_humans = tracked
        return tracked
        
class HumanTrackerWithVelocity(HumanTracker):

    def __init__(self, bus=1, addr=0x52, busy_pin=4):
        super().__init__(bus, addr, busy_pin)
        self.human_history = {}

    def get_height_from_floor(self, distance_mm):
        return SENSOR_TO_FLOOR_MM - distance_mm
        
    def calibrate_floor_distance(self, samples=20):
        distances = []

        print("Calibrating floor distance... Please ensure area is empty.")

        for _ in range(samples):
            humans = self.detect_humans()

            if humans:
                # Take the farthest detected object (likely floor)
                max_dist = max(h["distance_mm"] for h in humans)
                distances.append(max_dist)

            time.sleep(0.05)

        if not distances:
            raise RuntimeError("Calibration failed: no distance readings")

        # Use median for robustness against noise
        distances.sort()
        floor_distance = distances[len(distances) // 2]

        print(f"Calibrated floor distance: {floor_distance} mm")

        return floor_distance

    def compute_fall_confidence(self, h):

        # Normalize velocity (0 ? 1)
        velocity_score = min(max(h["velocity_mm_s"] / 2000, 0), 1)

        # Normalize height (closer to floor = higher score)
        height_score = min(max((FALL_HEIGHT_THRESHOLD - h["height_mm"]) / FALL_HEIGHT_THRESHOLD, 0), 1)

        # Normalize strength (optional tuning)
        strength_score = min(max((h["strength"] - HUMAN_STRENGTH_MIN) / (HUMAN_STRENGTH_MAX - HUMAN_STRENGTH_MIN), 0), 1)

        # Weighted sum
        confidence = (
            0.5 * velocity_score +
            0.3 * height_score +
            0.2 * strength_score
        )

        return confidence

    def track_humans_with_velocity(self):

        humans = self.detect_humans()
        tracked = []
        current_time = time.time()

        for h in humans:

            closest_prev = None
            min_dist = 10000

            for prev in self.previous_humans:
                d_diff = abs(prev["distance_mm"] - h["distance_mm"])

                if d_diff < min_dist:
                    min_dist = d_diff
                    closest_prev = prev

            if closest_prev and min_dist < 200:
                human_id = closest_prev["id"]
            else:
                human_id = id(h)

            velocity = 0

            if human_id in self.human_history:

                prev_record = self.human_history[human_id]
                dt = current_time - prev_record["time"]

                if dt > 0:
                    velocity = (h["distance_mm"] - prev_record["distance_mm"]) / dt

            self.human_history[human_id] = {
                "distance_mm": h["distance_mm"],
                "time": current_time
            }

            height = self.get_height_from_floor(h["distance_mm"])

            temp = {
                "id": human_id,
                "distance_mm": h["distance_mm"],
                "height_mm": height,
                "strength": h["strength"],
                "velocity_mm_s": velocity
            }

            temp["fall_confidence"] = self.compute_fall_confidence(temp)

            tracked.append(temp)

        self.previous_humans = tracked
        return tracked

    def detect_fall(self, tracked_humans):

        falls = []

        for h in tracked_humans:

            confidence = self.compute_fall_confidence(h)

            if confidence > FALL_CONFIDENCE_THRESHOLD:

                h["fall_confidence"] = confidence
                falls.append(h)

        return falls
        
        
if __name__ == "__main__":

    tracker = HumanTrackerWithVelocity(bus=1, addr=0x52, busy_pin=4)
    tracker.init()
    SENSOR_TO_FLOOR_MM = tracker.calibrate_floor_distance()

    tracker.set_start_and_end_range(
        HUMAN_DISTANCE_MIN_MM,
        HUMAN_DISTANCE_MAX_MM
    )

    while True:

        tracked = tracker.track_humans_with_velocity()

        if tracked:

            print("Tracked Humans:")

            for h in tracked:

                print(
                    f"ID: {h['id']} "
                    f"Dist: {h['distance_mm']} mm "
                    f"Height: {h['height_mm']} mm "
                    f"Vel: {h['velocity_mm_s']:.2f} mm/s "
                    f"Str: {h['strength']:.2f}"
                )

        else:
            print("No humans detected")

        falls = tracker.detect_fall(tracked)

        for f in falls:

            print(
                f"FALL DETECTED! "
                f"ID: {f['id']} "
                f"Confidence: {f['fall_confidence']:.2f} "
                f"Height: {f['height_mm']} mm "
                f"Velocity: {f['velocity_mm_s']:.2f} mm/s"
            )

        time.sleep(0.1)
