import sys

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe module is not installed. Please install it using 'pip install mediapipe'.")

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel,
        QVBoxLayout, QWidget, QComboBox, QPushButton, QHBoxLayout
    )
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import QTimer, Qt
except ImportError:
    raise ImportError("PyQt5 module is not installed. Please install it using 'pip install PyQt5'.")

try:
    import cv2
except ImportError:
    raise ImportError("cv2 module is not installed. Please install it using 'pip install opencv-python'.")

import pandas as pd
import numpy as np
from collections import deque
import pyqtgraph as pg

class ELANVideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Segment Viewer with ELAN Timeline")
        self.resize(1200, 850)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        top_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        
        self.eye_button = QPushButton("Load Eye Tracking Video")  # Text moved to button
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
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        right_layout.addWidget(self.reset_button)

        
        self.label_plot_hand = QLabel("Select Hand to Plot")
        right_layout.addWidget(self.label_plot_hand)
        self.hand_filter_dropdown = QComboBox()
        self.hand_filter_dropdown.addItems(['Both', 'Left', 'Right'])
        right_layout.addWidget(self.hand_filter_dropdown)

      
        video_layout = QHBoxLayout()
        self.video_label = QLabel("Main Video Preview")
        self.video_label.setFixedHeight(400)
        video_layout.addWidget(self.video_label)

        self.eyetrack_label = QLabel("Eye Tracking Preview")
        self.eyetrack_label.setFixedHeight(400)
        video_layout.addWidget(self.eyetrack_label)

        left_layout.addLayout(video_layout)

        self.graph_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_widget)
        self.buffers = [deque(maxlen=500) for _ in range(3)]  # V, X, Y
        self.curves = []
        labels = ["Velocity", "Mean X", "Mean Y"]
        for label in labels:
            plot = self.graph_widget.addPlot(title=label)
            plot.showGrid(x=True, y=True)
            curve = plot.plot(pen=pg.mkPen(color='y', width=2))
            self.curves.append(curve)
            self.graph_widget.nextRow()

        top_layout.addLayout(left_layout, 3)
        top_layout.addLayout(right_layout, 1)
        main_layout.addLayout(top_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

        self.fps = 30
        self.end_time = None
        self.video_loaded = False
        self.raw_data = []

      
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.prev_coords = None

    def load_elan_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Annotation File", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            df = pd.read_csv(path, sep='\t')
            self.elan_df = df
            self.modality_columns = [col for col in df.columns if col not in ['Anfangszeit - msec', 'Endzeit - msec']]
            self.modality_dropdown.clear()
            self.modality_dropdown.addItems(self.modality_columns)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .txt file: {str(e)}")

    def update_start_end_dropdowns(self):
        col = self.modality_dropdown.currentText()
        if not col or not hasattr(self, 'elan_df'):
            return
        entries = self.elan_df[col].dropna().astype(str).tolist()
        self.start_dropdown.clear()
        self.start_dropdown.addItems(entries)
        self.end_dropdown.clear()
        self.end_dropdown.addItems(entries)

    def ensure_end_options_filtered(self):
        start_index = self.start_dropdown.currentIndex()
        total = self.end_dropdown.count()
        if start_index >= 0 and start_index < total:
            current_entries = [self.end_dropdown.itemText(i) for i in range(total)]
            self.end_dropdown.clear()
            self.end_dropdown.addItems(current_entries[start_index:])

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.video_loaded = True
        QMessageBox.information(self, "Video Loaded", "Video loaded successfully. Select a timeline to begin.")

    def load_eyetrack_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Eye Tracking Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.eyetrack_cap = cv2.VideoCapture(path)
        self.eyetrack_fps = self.eyetrack_cap.get(cv2.CAP_PROP_FPS) or 30
        QMessageBox.information(self, "Eye Tracking Video Loaded", "Eye tracking video loaded successfully.")
    

    def play_selected_segment(self):
        if not hasattr(self, 'elan_df') or not self.video_loaded:
            QMessageBox.warning(self, "Missing Input", "Please load both video and annotation file.")
            return
        try:
            col = self.modality_dropdown.currentText()
            start_label = self.start_dropdown.currentText()
            end_label = self.end_dropdown.currentText()
            df = self.elan_df[self.elan_df[col].notna()]
            start_row = df[df[col].astype(str) == start_label].iloc[0]
            end_row = df[df[col].astype(str) == end_label].iloc[0]
            self.start_time = float(start_row['Anfangszeit - msec'])
            self.end_time = float(end_row['Endzeit - msec'])
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
            self.timer.start(int(1000 / self.fps))
            self.duration_label.setText(f"Duration: {int(self.end_time - self.start_time)} ms")
            self.raw_data = []
            self.prev_coords = None
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", str(e))

    def set_manual_time_range(self):
        if not hasattr(self, 'elan_df'):
            return
        col = self.modality_dropdown.currentText()
        start_label = self.start_dropdown.currentText()
        end_label = self.end_dropdown.currentText()
        try:
            df = self.elan_df[self.elan_df[col].notna()]
            start_row = df[df[col].astype(str) == start_label].iloc[0]
            end_row = df[df[col].astype(str) == end_label].iloc[0]
            self.start_time = float(start_row['Anfangszeit - msec'])
            self.end_time = float(end_row['Endzeit - msec'])
            self.duration_label.setText(f"Duration: {int(self.end_time - self.start_time)} ms")
        except Exception as e:
            print("Error setting manual time range:", e)

    def update(self):
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



        eyetrack_offset = -32123  # adjust this as needed (in ms)
        adjusted_time = max(0, current_time + eyetrack_offset)

        if hasattr(self, 'eyetrack_cap') and self.eyetrack_cap.isOpened():
            self.eyetrack_cap.set(cv2.CAP_PROP_POS_MSEC, adjusted_time)
            ret2, eye_frame = self.eyetrack_cap.read()
            if ret2:
                rgb_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
                qt_eye_img = QImage(rgb_eye.data, rgb_eye.shape[1], rgb_eye.shape[0], QImage.Format_RGB888)
                self.eyetrack_label.setPixmap(QPixmap.fromImage(qt_eye_img).scaled(
                    self.eyetrack_label.width(), self.eyetrack_label.height(), Qt.KeepAspectRatio))

        # MediaPipe Hand Tracking
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        coords = []
        selected_filter = self.hand_filter_dropdown.currentText().lower()
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                if selected_filter == 'both' or selected_filter == hand_label:
                    for lm in hand_landmarks.landmark:
                        coords.append((lm.x * width, lm.y * height))
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if coords:
            x_vals = [x for x, _ in coords]
            y_vals = [y for _, y in coords]
            mean_x = np.mean(x_vals)
            mean_y = np.mean(y_vals)

            if self.prev_coords is not None:
                dx = mean_x - self.prev_coords[0]
                dy = mean_y - self.prev_coords[1]
                velocity = np.sqrt(dx**2 + dy**2) * self.fps
            else:
                velocity = 0.0
            self.prev_coords = (mean_x, mean_y)
        else:
            velocity = 0.0
            mean_x = 0.0
            mean_y = 0.0

        for i, val in enumerate([velocity, mean_x, mean_y]):
            self.buffers[i].append(val)
            self.curves[i].setData(list(range(len(self.buffers[i]))), list(self.buffers[i]))
            if i == 0:
                self.raw_data.append([
                    int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    int(self.cap.get(cv2.CAP_PROP_POS_MSEC)),
                    velocity, mean_x, mean_y
                ])

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def export_segment_data(self):
        if not hasattr(self, 'raw_data') or not self.raw_data:
            QMessageBox.warning(self, "No Data", "No motion data to export.")
            return
        try:
            df = pd.DataFrame(self.raw_data, columns=["Frame", "Time (ms)", "Velocity", "Mean X", "Mean Y"])
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Exported", f"Segment data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def reset_view(self):
        self.buffers = [deque(maxlen=500) for _ in range(3)]
        for curve in self.curves:
            curve.clear()
        self.video_label.clear()
        self.eyetrack_label.clear()
        self.duration_label.setText("Duration: -- ms")
        self.raw_data = []
        if hasattr(self, 'cap'):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        self.timer.stop()
        self.prev_coords = None

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ELANVideoPlayer()
    window.show()
    sys.exit(app.exec_())
