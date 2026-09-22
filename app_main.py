import os

os.environ["ULTRALYTICS_AUTO_UPDATE"] = "False"

import sys
import time
from pathlib import Path

import cv2
import pygame
from PySide6.QtCore import QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget,
)
from ultralytics import YOLO

from sea_sentinel import BoatIDManager, DetectionEventLogger, ensure_model, resolve_target_classes

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ensure_model(BASE_DIR / "海面船只检测.pt")
OUTPUT_DIR = BASE_DIR / "outputs"


class DisplayWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#101820;border:1px solid #324451;color:#9fb0bc")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(QSize(480, 300))
        self.setText("请选择视频、图片或摄像头")
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self):
        if self._pixmap and not self._pixmap.isNull():
            super().setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MediaWorker(QThread):
    frame_processed = Signal(QImage)
    metrics_updated = Signal(int, float, float)
    error_occurred = Signal(str)
    prediction_stopped = Signal()
    alarm_changed = Signal(bool)

    def __init__(self, source, model=None, is_camera=False, confidence=0.35, iou=0.5):
        super().__init__()
        self.source, self.model, self.is_camera = source, model, is_camera
        self.confidence, self.iou = confidence, iou
        self.running, self.is_predicting, self.stop_requested = True, False, False
        self.boat_detected = False
        self.warning_sound = self.alarm_sound = None
        self.id_manager = BoatIDManager()
        self.target_classes = resolve_target_classes(model.names) if model else None
        self.logger = DetectionEventLogger(OUTPUT_DIR / "events.csv")

    def set_model(self, model):
        self.model = model
        self.target_classes = resolve_target_classes(model.names)

    def set_thresholds(self, confidence, iou):
        self.confidence, self.iou = confidence, iou

    def _load_sounds(self):
        if self.warning_sound:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.warning_sound = pygame.mixer.Sound(str(BASE_DIR / "警告.mp3"))
            self.alarm_sound = pygame.mixer.Sound(str(BASE_DIR / "警报音效.mp3"))
        except Exception as exc:
            print(f"Audio disabled: {exc}")

    def _set_alarm(self, active, count=0, confidence=0.0):
        if active == self.boat_detected:
            return
        self.boat_detected = active
        self.logger.write("entered" if active else "cleared", str(self.source), count, confidence)
        if active:
            self._load_sounds()
            if self.warning_sound:
                self.warning_sound.play(loops=2)
            if self.alarm_sound:
                self.alarm_sound.play(loops=-1)
        else:
            if self.warning_sound:
                self.warning_sound.stop()
            if self.alarm_sound:
                self.alarm_sound.stop()
        self.alarm_changed.emit(active)

    def set_prediction_mode(self, enabled):
        self.is_predicting = enabled
        if not enabled:
            self._set_alarm(False)

    def request_stop_prediction(self):
        self.stop_requested = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开媒体源: {self.source}")
            return
        try:
            while self.running:
                if self.stop_requested:
                    self.set_prediction_mode(False)
                    self.stop_requested = False
                    self.prediction_stopped.emit()
                ok, frame = cap.read()
                if not ok:
                    if self.is_camera:
                        self.error_occurred.emit("摄像头连接中断")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                count, maximum, started = 0, 0.0, time.perf_counter()
                if self.is_predicting and self.model:
                    try:
                        result = self.model.track(
                            frame, classes=self.target_classes, conf=self.confidence,
                            iou=self.iou, persist=True, verbose=False,
                        )[0]
                        boxes, active_ids = result.boxes, []
                        if boxes is not None:
                            coordinates = boxes.xyxy.cpu().numpy()
                            confidences = boxes.conf.cpu().numpy()
                            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
                            count = len(coordinates)
                            maximum = float(confidences.max()) if count else 0.0
                            for index, box in enumerate(coordinates):
                                x1, y1, x2, y2 = map(int, box)
                                track_id = int(track_ids[index]) if track_ids is not None else None
                                if track_id is None:
                                    label = f"Boat {confidences[index]:.0%}"
                                else:
                                    active_ids.append(track_id)
                                    label = f"Boat #{self.id_manager.get_display_id(track_id)} {confidences[index]:.0%}"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
                                cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 210, 255), 2)
                        self.id_manager.update_active_track_ids(active_ids)
                        self._set_alarm(count > 0, count, maximum)
                    except Exception as exc:
                        self.error_occurred.emit(f"推理失败: {exc}")
                        self.set_prediction_mode(False)
                fps = 1.0 / max(time.perf_counter() - started, 1e-6) if self.is_predicting else 0.0
                self.metrics_updated.emit(count, maximum, fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                self.frame_processed.emit(QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy())
                if not self.is_predicting:
                    self.msleep(15)
        finally:
            self._set_alarm(False)
            cap.release()

    def stop(self):
        self.running = False
        self.wait(3000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海面哨兵 - 船只智能监测系统")
        self.resize(1080, 720)
        self.model = self.media_worker = self.media_source = None
        self.current_media_type = self.current_image_bgr = self.last_frame = None
        self._setup_ui()
        if DEFAULT_MODEL.exists():
            self._load_model(DEFAULT_MODEL)

    def _setup_ui(self):
        central, panel = QWidget(), QFrame()
        self.setCentralWidget(central)
        layout, panel_layout = QVBoxLayout(central), QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        splitter, self.display_label = QSplitter(Qt.Vertical), DisplayWidget()
        splitter.addWidget(self.display_label)
        controls = QHBoxLayout()
        self.btn_upload = QPushButton("打开媒体")
        menu = QMenu(self)
        file_action, camera_action = QAction("图片或视频", self), QAction("摄像头", self)
        file_action.triggered.connect(self.open_file_dialog)
        camera_action.triggered.connect(self.open_camera)
        menu.addActions([file_action, camera_action])
        self.btn_upload.setMenu(menu)
        self.btn_model, self.btn_predict = QPushButton("加载模型"), QPushButton("开始监测")
        self.btn_stop, self.btn_snapshot = QPushButton("停止监测"), QPushButton("保存截图")
        self.btn_model.clicked.connect(self.import_model)
        self.btn_predict.clicked.connect(self.start_prediction)
        self.btn_stop.clicked.connect(self.stop_prediction)
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        self.btn_predict.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        for button in (self.btn_upload, self.btn_model, self.btn_predict, self.btn_stop, self.btn_snapshot):
            button.setMinimumHeight(42)
            controls.addWidget(button)
        thresholds = QHBoxLayout()
        self.confidence_input, self.iou_input = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, low, high, value in ((self.confidence_input, .05, .95, .35), (self.iou_input, .1, .9, .5)):
            widget.setRange(low, high)
            widget.setSingleStep(.05)
            widget.setValue(value)
        thresholds.addWidget(QLabel("置信度")); thresholds.addWidget(self.confidence_input)
        thresholds.addWidget(QLabel("IoU")); thresholds.addWidget(self.iou_input); thresholds.addStretch()
        self.metrics_label = QLabel("目标 0 | 最高置信度 0% | 推理 0.0 FPS")
        thresholds.addWidget(self.metrics_label)
        status = QHBoxLayout()
        self.lbl_status, self.progress_bar = QLabel("就绪"), QProgressBar()
        self.progress_bar.setRange(0, 0); self.progress_bar.hide()
        self.warning_label = QLabel("告警：监测区域发现船只")
        self.warning_label.setStyleSheet("color:#d93025;font-size:16px;font-weight:700")
        self.warning_label.hide()
        status.addWidget(self.lbl_status); status.addWidget(self.progress_bar); status.addStretch(); status.addWidget(self.warning_label)
        panel_layout.addLayout(controls); panel_layout.addLayout(thresholds); panel_layout.addLayout(status)
        splitter.addWidget(panel); splitter.setStretchFactor(0, 7); splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择媒体", str(BASE_DIR / "test_video"), "媒体文件 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mkv *.mov)")
        if path:
            self.load_media(path)

    def open_camera(self):
        self.load_media(0, True)

    @staticmethod
    def determine_file_type(path):
        if isinstance(path, int): return "camera"
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}: return "image"
        if Path(path).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}: return "video"
        return "unknown"

    def load_media(self, source, is_camera=False):
        if self.media_worker:
            self.media_worker.stop()
        self.media_source = source
        self.current_media_type = "camera" if is_camera else self.determine_file_type(source)
        if self.current_media_type == "image":
            self.current_image_bgr = cv2.imread(source)
            if self.current_image_bgr is None:
                return self.handle_error("无法读取图片")
            self._show_bgr_frame(self.current_image_bgr)
            self.lbl_status.setText(f"已加载图片: {Path(source).name}")
        elif self.current_media_type in {"video", "camera"}:
            self.media_worker = MediaWorker(source, self.model, is_camera, self.confidence_input.value(), self.iou_input.value())
            self.media_worker.frame_processed.connect(self.update_video_frame)
            self.media_worker.metrics_updated.connect(self.update_metrics)
            self.media_worker.error_occurred.connect(self.handle_error)
            self.media_worker.prediction_stopped.connect(self.on_prediction_stopped)
            self.media_worker.alarm_changed.connect(self.warning_label.setVisible)
            self.media_worker.start()
            self.lbl_status.setText("视频流已就绪")
        else:
            return self.handle_error("不支持的媒体格式")
        self._update_buttons()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", str(BASE_DIR), "YOLO 模型 (*.pt *.onnx)")
        if path:
            self._load_model(Path(path))

    def _load_model(self, path):
        try:
            self.progress_bar.show(); QApplication.processEvents()
            self.model = YOLO(str(path))
            if self.media_worker:
                self.media_worker.set_model(self.model)
            self.lbl_status.setText(f"模型已加载: {path.name}")
            self._update_buttons()
        except Exception as exc:
            self.handle_error(f"模型加载失败: {exc}")
        finally:
            self.progress_bar.hide()

    def _update_buttons(self):
        self.btn_predict.setEnabled(self.model is not None and self.media_source is not None)
        self.btn_snapshot.setEnabled(self.last_frame is not None)

    def start_prediction(self):
        if not self.model:
            return
        confidence, iou = self.confidence_input.value(), self.iou_input.value()
        if self.current_media_type == "image":
            try:
                result = self.model(self.current_image_bgr, classes=resolve_target_classes(self.model.names), conf=confidence, iou=iou, verbose=False)[0]
                self._show_bgr_frame(result.plot())
                count = len(result.boxes) if result.boxes is not None else 0
                maximum = float(result.boxes.conf.max()) if count else 0.0
                self.update_metrics(count, maximum, 0.0)
                self.warning_label.setVisible(count > 0)
                self.lbl_status.setText("图片检测完成")
            except Exception as exc:
                self.handle_error(f"推理失败: {exc}")
            return
        self.media_worker.set_thresholds(confidence, iou)
        self.media_worker.set_prediction_mode(True)
        self.btn_predict.setEnabled(False); self.btn_stop.setEnabled(True)
        self.lbl_status.setText("正在实时监测")

    def stop_prediction(self):
        if self.media_worker:
            self.btn_stop.setEnabled(False)
            self.media_worker.request_stop_prediction()

    @Slot()
    def on_prediction_stopped(self):
        self.lbl_status.setText("监测已停止，视频继续预览")
        self.btn_stop.setEnabled(False); self.btn_predict.setEnabled(True); self.warning_label.hide()

    @Slot(int, float, float)
    def update_metrics(self, count, confidence, fps):
        self.metrics_label.setText(f"目标 {count} | 最高置信度 {confidence:.0%} | 推理 {fps:.1f} FPS")

    @Slot(QImage)
    def update_video_frame(self, image):
        self.last_frame = image.copy()
        self.display_label.setPixmap(QPixmap.fromImage(image))
        self.btn_snapshot.setEnabled(True)

    def _show_bgr_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self.update_video_frame(QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy())

    def save_snapshot(self):
        if self.last_frame is None:
            return
        directory = OUTPUT_DIR / "snapshots"; directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        self.lbl_status.setText(f"截图已保存: {path.name}" if self.last_frame.save(str(path), "JPG", 92) else "截图保存失败")

    @Slot(str)
    def handle_error(self, message):
        self.lbl_status.setText("发生错误")
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        if self.media_worker:
            self.media_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
