import time
from sensors.camera import Camera
from sensors.mmWave import HumanTrackerWithVelocity
from sensors.microphone import Microphone

from inference.pose_detection import PoseEstimator
from inference.weighted_fusion import WeightedFusion

radar = HumanTrackerWithVelocity(bus=1, addr=0x52, busy_pin=4)  # mmWave radar initialization
camera = Camera()  # Initialize camera module
mic = Microphone()  # Initialize Microphone

pose_model = PoseEstimator()  # Load pose estimation model (MoveNet)
weighted_fusion = WeightedFusion() # Load weighted fusion

FUSION_FALL_THRESHOLD = 0.7

HUMAN_DISTANCE_MIN_MM = 0
HUMAN_DISTANCE_MAX_MM = 3000

HUMAN_STRENGTH_MIN = 0
HUMAN_STRENGTH_MAX = 10

SENSOR_TO_FLOOR_MM = radar.calibrate_floor_distance()

radar.set_start_and_end_range(
	HUMAN_DISTANCE_MIN_MM,
	HUMAN_DISTANCE_MAX_MM
)

print("SYSTEM STARTED")

try:
    mic.start()
    radar.init()

    while True:
        mic_confidence = mic.get_confidence()
        
        tracked = radar.track_humans_with_velocity
        
        # Skip iteration if no human to track
        if len(tracked) == 0:
            continue
        
        falls = radar.detect_fall(tracked)
        
        # Get first fall confidence
        radar_fall_confidence = falls[0]['fall_confidence']

        # Capture frame from webcam
        frame = camera.get_frame()

        # If frame capture failed, skip this loop iteration
        if frame is None:
            continue

        # Check if a person is visible using motion detection
        person_visible = camera.detect_person(frame)

        # If no person detected in camera view, skip pose estimation
        if not person_visible:
            camera_fall_confidence = None
        else:
			# Get frame dimensions for coordinate scaling
            frame_height, frame_width, _ = frame.shape

            # Run pose estimation model on frame to detect body keypoints
            keypoints = pose_model.estimate_pose(frame)
            
			# Determine whether the detected pose indicates a fall
            camera_fall_confidence = pose_model.detect_fall_pose(keypoints, frame_width, frame_height)
        
        final_fall_score = weighted_fusion.fuse(camera_conf=camera_fall_confidence,
        mmwave_conf=radar_fall_confidence, mic_conf=mic_confidence)
        
        print(f"Camera: {camera_fall_confidence}, Radar: {radar_fall_confidence}, Mic: {mic_confidence}")
        print(f"Final fall conf score: {final_fall_score}")
        
        if final_fall_score > FUSION_FALL_THRESHOLD:
            print(f"[FUSION] POSSIBLE FALL DETECTED")

        # Small delay to reduce CPU usage and stabilize processing
        time.sleep(0.2)

except KeyboardInterrupt:
    # Stop System
    print("STOPPING...")
    camera.release()
    mic.stop()

