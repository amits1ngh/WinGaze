try:
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt
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


def frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, QImage.Format_RGB888)
    return qimage.copy()


def set_label_image(label, frame) -> None:
    qimage = frame_to_qimage(frame)
    label.setPixmap(
        QPixmap.fromImage(qimage).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio
        )
    )
