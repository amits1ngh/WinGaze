import os

try:
    from PyQt5.QtWidgets import (
        QMainWindow,
        QFileDialog,
        QMessageBox,
        QLabel,
        QVBoxLayout,
        QWidget,
        QComboBox,
        QPushButton,
    )
    from PyQt5.QtCore import QTimer
except ImportError as exc:
    raise ImportError(
        "PyQt5 module is not installed. Please install it using 'pip install PyQt5'."
    ) from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "cv2 module is not installed. Please install it using 'pip install opencv-python'."
    ) from exc

from config.settings import AppConfig
from core.hand_tracking import HandTracker
from data_io.elan_reader import read_elan_file, get_segment_times, extract_text_annotations
from data_io.fixation_reader import read_fixations_csv
from data_io.exporter import export_tracking_data
from vis.rerun_logger import RerunLogger


class ELANVideoPlayer(QMainWindow):
    def __init__(self, config: AppConfig, rerun_logger: RerunLogger) -> None:
        super().__init__()
        self.config = config
        self.rerun_logger = rerun_logger

        self.setWindowTitle(self.config.window_title)
        self.resize(self.config.window_width, self.config.window_height)

        self._setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.fps = self.config.default_fps
        self.end_time = None
        self.start_time = None
        self.video_loaded = False
        self.raw_data = []
        self.elan_df = None
        self.speech_annotations = []
        self.speech_columns = []
        self.speech_speaker_order = []

        self.hand_tracker = HandTracker(
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_segmentation=self.config.mask_participant,
        )

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, path)

    def _load_aoi_fixations(self):
        fixations = []
        sources = [
            ("Robot", self.config.fixation_robot_path),
            ("Task", self.config.fixation_task_path),
        ]
        for label, path in sources:
            resolved = self._resolve_path(path)
            if not os.path.exists(resolved):
                continue
            try:
                fixations.extend(
                    read_fixations_csv(
                        resolved,
                        label,
                        time_col=self.config.fixation_time_col,
                        start_col=self.config.fixation_start_col,
                        duration_col=self.config.fixation_duration_col,
                        time_scale=self.config.fixation_time_scale,
                        duration_scale=self.config.fixation_duration_scale,
                        fixation_id_col=self.config.fixation_id_col,
                        dedupe=self.config.fixation_dedupe,
                    )
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Fixation Load Warning",
                    f"Failed to load {resolved}: {exc}",
                )
        return fixations

    def _setup_ui(self) -> None:
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        right_layout = QVBoxLayout()

        self.eye_button = QPushButton("Load Eye Tracking Video")
        self.eye_button.clicked.connect(self.load_eyetrack_video)
        right_layout.addWidget(self.eye_button)

        self.play_button = QPushButton("Select Video File")
        self.play_button.clicked.connect(self.load_video)
        right_layout.addWidget(self.play_button)

        self.elan_button = QPushButton("Select Annotation File")
        self.elan_button.clicked.connect(self.load_elan_file)
        right_layout.addWidget(self.elan_button)

        self.label_column = QLabel("Select Multimodal Feature")
        right_layout.addWidget(self.label_column)
        self.modality_dropdown = QComboBox()
        self.modality_dropdown.currentIndexChanged.connect(self.update_start_end_dropdowns)
        right_layout.addWidget(self.modality_dropdown)

        self.label_start = QLabel("Select Start Time")
        right_layout.addWidget(self.label_start)
        self.start_dropdown = QComboBox()
        self.start_dropdown.currentIndexChanged.connect(self.ensure_end_options_filtered)
        right_layout.addWidget(self.start_dropdown)

        self.label_end = QLabel("Select End Time")
        right_layout.addWidget(self.label_end)
        self.end_dropdown = QComboBox()
        self.end_dropdown.currentIndexChanged.connect(self.set_manual_time_range)
        right_layout.addWidget(self.end_dropdown)

        self.play_segment_button = QPushButton("Play Segment")
        self.play_segment_button.clicked.connect(self.play_selected_segment)
        right_layout.addWidget(self.play_segment_button)

        self.duration_label = QLabel("Duration: -- ms")
        right_layout.addWidget(self.duration_label)
        self.export_button = QPushButton("Export Segment Data to CSV")
        self.export_button.clicked.connect(self.export_segment_data)
        right_layout.addWidget(self.export_button)

        self.label_plot_hand = QLabel("Select Hand to Plot")
        right_layout.addWidget(self.label_plot_hand)
        self.hand_filter_dropdown = QComboBox()
        self.hand_filter_dropdown.addItems(["Both", "Left", "Right"])
        right_layout.addWidget(self.hand_filter_dropdown)
        main_layout.addLayout(right_layout)

    def load_elan_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", "", "Text Files (*.txt)"
        )
        if not path:
            return
        try:
            df, modality_columns = read_elan_file(path)
            self.elan_df = df
            self.modality_columns = modality_columns
            self.modality_dropdown.clear()
            self.modality_dropdown.addItems(self.modality_columns)
            self.speech_columns = [
                col
                for col in df.columns
                if any(
                    key in str(col).strip().lower() for key in ("participant", "robot")
                )
            ]
            self.speech_speaker_order = [str(col).strip() for col in self.speech_columns]
            self.speech_annotations = extract_text_annotations(df, self.speech_columns)
            self._mark_loaded(self.elan_button)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load .txt file: {exc}")

    def update_start_end_dropdowns(self) -> None:
        col = self.modality_dropdown.currentText()
        if not col or self.elan_df is None:
            return
        entries = self.elan_df[col].dropna().astype(str).tolist()
        self.start_dropdown.clear()
        self.start_dropdown.addItems(entries)
        self.end_dropdown.clear()
        self.end_dropdown.addItems(entries)

    def ensure_end_options_filtered(self) -> None:
        start_index = self.start_dropdown.currentIndex()
        total = self.end_dropdown.count()
        if 0 <= start_index < total:
            current_entries = [self.end_dropdown.itemText(i) for i in range(total)]
            self.end_dropdown.clear()
            self.end_dropdown.addItems(current_entries[start_index:])

    def load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.config.default_fps
        self.video_loaded = True
        self._mark_loaded(self.play_button)

    def load_eyetrack_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Eye Tracking Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.eyetrack_cap = cv2.VideoCapture(path)
        self.eyetrack_fps = self.eyetrack_cap.get(cv2.CAP_PROP_FPS) or self.config.default_fps
        self._mark_loaded(self.eye_button)

    @staticmethod
    def _mark_loaded(button: QPushButton) -> None:
        button.setStyleSheet("QPushButton { background-color: #2e7d32; color: #ffffff; }")

    def play_selected_segment(self) -> None:
        if self.elan_df is None or not self.video_loaded:
            QMessageBox.warning(
                self, "Missing Input", "Please load both video and annotation file."
            )
            return
        try:
            col = self.modality_dropdown.currentText()
            start_label = self.start_dropdown.currentText()
            end_label = self.end_dropdown.currentText()
            segment = get_segment_times(self.elan_df, col, start_label, end_label)
            self.start_time = segment.start_ms
            self.end_time = segment.end_ms
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
            interval = int(1000 / self.fps) if self.fps else int(1000 / self.config.default_fps)
            self.timer.start(interval)
            self.duration_label.setText(
                f"Duration: {int(self.end_time - self.start_time)} ms"
            )
            self.raw_data = []
            self.hand_tracker.reset()
            if self.rerun_logger:
                self.rerun_logger.start_segment(self.start_time)
                self.rerun_logger.set_speech_annotations(
                    self.speech_annotations,
                    speaker_order=self.speech_speaker_order,
                    segment_start_ms=self.start_time,
                    segment_end_ms=self.end_time,
                )
                fixations = self._load_aoi_fixations()
                if fixations:
                    self.rerun_logger.set_aoi_probabilities(
                        fixations,
                        bin_ms=self.config.rerun.aoi_bin_ms,
                        smooth_kind=self.config.rerun.aoi_smooth_kind,
                        smooth_ms=self.config.rerun.aoi_smooth_ms,
                        include_none=self.config.rerun.aoi_include_none,
                        segment_start_ms=self.start_time,
                        segment_end_ms=self.end_time,
                        eyetrack_offset_ms=self.config.eyetrack_offset_ms,
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))

    def set_manual_time_range(self) -> None:
        if self.elan_df is None:
            return
        col = self.modality_dropdown.currentText()
        start_label = self.start_dropdown.currentText()
        end_label = self.end_dropdown.currentText()
        try:
            segment = get_segment_times(self.elan_df, col, start_label, end_label)
            self.start_time = segment.start_ms
            self.end_time = segment.end_ms
            self.duration_label.setText(
                f"Duration: {int(self.end_time - self.start_time)} ms"
            )
        except Exception as exc:
            print("Error setting manual time range:", exc)

    def update_frame(self) -> None:
        if not self.video_loaded:
            return
        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if self.end_time and current_time > self.end_time:
            self.timer.stop()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        adjusted_time = max(0, current_time + self.config.eyetrack_offset_ms)

        eye_frame = None
        if hasattr(self, "eyetrack_cap") and self.eyetrack_cap.isOpened():
            self.eyetrack_cap.set(cv2.CAP_PROP_POS_MSEC, adjusted_time)
            ret2, eye_frame = self.eyetrack_cap.read()
            if not ret2:
                eye_frame = None

        metrics = self.hand_tracker.process_frame(
            frame, self.hand_filter_dropdown.currentText(), self.fps, self.config.mask_participant
        )

        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self.raw_data.append(
            [frame_id, time_ms, metrics.velocity, metrics.mean_x, metrics.mean_y]
        )

        if self.rerun_logger:
            self.rerun_logger.log_frame(
                frame_id, time_ms, metrics, frame, eye_frame, fps=self.fps
            )

    def export_segment_data(self) -> None:
        if not self.raw_data:
            QMessageBox.warning(self, "No Data", "No motion data to export.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", "", "CSV Files (*.csv)"
            )
            if path:
                export_tracking_data(self.raw_data, path)
                QMessageBox.information(self, "Exported", f"Segment data saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def reset_view(self) -> None:
        self.duration_label.setText("Duration: -- ms")
        self.raw_data = []
        if hasattr(self, "cap"):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        self.timer.stop()
        self.hand_tracker.reset()
        if self.rerun_logger:
            self.rerun_logger.reset()

    def closeEvent(self, event) -> None:
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "eyetrack_cap") and self.eyetrack_cap.isOpened():
            self.eyetrack_cap.release()
        self.hand_tracker.close()
        super().closeEvent(event)
