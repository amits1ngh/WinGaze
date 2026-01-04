try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe module is not installed. Please install it using 'pip install mediapipe'."
    ) from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "cv2 module is not installed. Please install it using 'pip install opencv-python'."
    ) from exc

import numpy as np

from core.data_types import TrackingMetrics


class HandTracker:
    def __init__(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        enable_segmentation: bool = False,
    ) -> None:
        self._mp_pose = mp.solutions.pose
        self._drawing_utils = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=True,
        )
        self._prev_coords = None
        self._visibility_threshold = 0.5
        self._segmentation_threshold = 0.5

    def reset(self) -> None:
        self._prev_coords = None

    def close(self) -> None:
        if hasattr(self._pose, "close"):
            self._pose.close()

    def process_frame(self, frame, hand_filter: str, fps: float, mask_participant: bool) -> TrackingMetrics:
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)
        if mask_participant:
            self._apply_segmentation_mask(frame, results)

        coords = []
        selected_filter = hand_filter.lower()
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST]
            if selected_filter in ("both", "left"):
                if getattr(left_wrist, "visibility", 1.0) >= self._visibility_threshold:
                    coords.append((left_wrist.x * width, left_wrist.y * height))
            if selected_filter in ("both", "right"):
                if getattr(right_wrist, "visibility", 1.0) >= self._visibility_threshold:
                    coords.append((right_wrist.x * width, right_wrist.y * height))
            self._drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self._mp_pose.POSE_CONNECTIONS
            )

        if coords:
            x_vals = [x for x, _ in coords]
            y_vals = [y for _, y in coords]
            mean_x = float(np.mean(x_vals))
            mean_y = float(np.mean(y_vals))

            if self._prev_coords is not None:
                dx = mean_x - self._prev_coords[0]
                dy = mean_y - self._prev_coords[1]
                velocity = float(np.sqrt(dx**2 + dy**2) * fps)
            else:
                velocity = 0.0
            self._prev_coords = (mean_x, mean_y)
        else:
            velocity = 0.0
            mean_x = float("nan")
            mean_y = float("nan")

        return TrackingMetrics(velocity=velocity, mean_x=mean_x, mean_y=mean_y)

    def _apply_segmentation_mask(self, frame, results) -> None:
        mask = getattr(results, "segmentation_mask", None)
        if mask is None:
            return
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        person = mask > self._segmentation_threshold
        frame[person] = 0
