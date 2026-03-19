import cv2


class Camera:
    def __init__(self, resolution=(320, 240), hog_interval=5):
        self.cap = cv2.VideoCapture(0)
        self.resolution = resolution
        self.hog_interval = hog_interval
        self.hog_frame_count = 0

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=False
        )

        # HOG person detector for far/stationary people
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Morphological kernel to clean up fg mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def _adaptive_threshold(self, contour, frame_height):
        """Scale contour threshold by vertical position — further = smaller threshold."""
        x, y, w, h = cv2.boundingRect(contour)
        center_y = y + h / 2
        return 1000 + (4000 * (center_y / frame_height))

    def _motion_detect(self, frame_small):
        """Background subtraction with morphological cleanup."""
        fg_mask = self.bg_subtractor.apply(frame_small)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Fill gaps in mask to merge nearby blobs
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_height = frame_small.shape[0]
        for c in contours:
            if cv2.contourArea(c) > self._adaptive_threshold(c, frame_height):
                return True
        return False

    def _hog_detect(self, frame_small):
        """HOG pedestrian detection — handles far/stationary people."""
        rects, weights = self.hog.detectMultiScale(
            frame_small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        # Filter weak detections
        if len(weights) > 0 and max(weights) > 0.3:
            return True
        return False

    def detect_person(self, frame):
        frame_small = cv2.resize(frame, self.resolution)

        # Fast path — motion detection
        if self._motion_detect(frame_small):
            # print("[CAMERA] HUMAN DETECTED (motion)")
            return True

        # Slow path — HOG every N frames for stationary/far people
        self.hog_frame_count += 1
        if self.hog_frame_count % self.hog_interval == 0:
            if self._hog_detect(frame_small):
                # print("[CAMERA] HUMAN DETECTED (HOG)")
                return True

        return False

    def release(self):
        self.cap.release()
