class WeightedFusion:
    def __init__(self):
        # Base reliability weights
        self.weights = {
            "camera": 0.5,
            "mmwave": 0.3,
            "mic": 0.2
        }

    def fuse(self, camera_conf=None, mmwave_conf=None, mic_conf=None):
        """
        Returns final fused confidence score (0 to 1)

        Inputs:
        - camera_conf: float (0-1) or None if unavailable
        - mmwave_conf: float (0-1) or None if unavailable
        - mic_conf: float (0-1) or None if unavailable
        """

        sensor_values = {
            "camera": camera_conf,
            "mmwave": mmwave_conf,
            "mic": mic_conf
        }

        active_sensors = {}
        active_weights = {}

        # -----------------------------------------
        # Step 1: Filter available sensors
        # -----------------------------------------
        for sensor, value in sensor_values.items():
            if value is not None:
                active_sensors[sensor] = value
                active_weights[sensor] = self.weights[sensor]

        # No sensors available
        if not active_sensors:
            return 0.0

        # -----------------------------------------
        # Step 2: Normalize weights
        # -----------------------------------------
        total_weight = sum(active_weights.values())

        for sensor in active_weights:
            active_weights[sensor] /= total_weight

        # -----------------------------------------
        # Step 3: Compute weighted sum
        # -----------------------------------------
        final_score = 0.0

        for sensor in active_sensors:
            final_score += active_sensors[sensor] * active_weights[sensor]

        return final_score
        
