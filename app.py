"""Hugging Face Spaces entry point for Sea Sentinel."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from threading import Lock

import cv2
import gradio as gr
from ultralytics import YOLO

from sea_sentinel import ensure_model, resolve_target_classes


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = ensure_model(BASE_DIR / "海面船只检测.pt")
MODEL = YOLO(str(MODEL_PATH))
MODEL_LOCK = Lock()


def detect_image(image, confidence: float, iou: float):
    if image is None:
        raise gr.Error("请先上传图片")
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    started = time.perf_counter()
    with MODEL_LOCK:
        result = MODEL.predict(
            frame,
            conf=confidence,
            iou=iou,
            classes=resolve_target_classes(MODEL.names),
            verbose=False,
        )[0]
    elapsed = time.perf_counter() - started
    count = len(result.boxes) if result.boxes is not None else 0
    maximum = float(result.boxes.conf.max()) if count else 0.0
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    summary = f"检测目标：{count} | 最高置信度：{maximum:.1%} | 耗时：{elapsed * 1000:.0f} ms"
    return annotated, summary


def detect_video(video_path: str, confidence: float, iou: float, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("请先上传视频")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error("无法读取上传的视频")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_file = tempfile.NamedTemporaryFile(prefix="sea_sentinel_", suffix=".mp4", delete=False)
    output_file.close()
    writer = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise gr.Error("无法创建结果视频")

    frame_index = detections = peak_count = 0
    started = time.perf_counter()
    try:
        with MODEL_LOCK:
            MODEL.predictor = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = MODEL.track(
                    frame,
                    conf=confidence,
                    iou=iou,
                    classes=resolve_target_classes(MODEL.names),
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )[0]
                count = len(result.boxes) if result.boxes is not None else 0
                detections += count
                peak_count = max(peak_count, count)
                writer.write(result.plot())
                frame_index += 1
                if total_frames and frame_index % 10 == 0:
                    progress(min(frame_index / total_frames, 1.0), desc="正在检测")
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    processing_fps = frame_index / elapsed if elapsed else 0.0
    summary = (
        f"处理帧数：{frame_index} | 累计检测框：{detections} | "
        f"单帧最多目标：{peak_count} | 处理速度：{processing_fps:.1f} FPS"
    )
    return output_file.name, summary


with gr.Blocks(title="海面哨兵") as demo:
    gr.Markdown("# 海面哨兵 · 船只智能监测")
    with gr.Tabs():
        with gr.Tab("图片检测"):
            with gr.Row():
                image_input = gr.Image(label="上传图片", type="numpy")
                image_output = gr.Image(label="检测结果", type="numpy")
            with gr.Row():
                image_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="置信度")
                image_iou = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="IoU")
            image_button = gr.Button("开始检测", variant="primary")
            image_summary = gr.Textbox(label="运行统计", interactive=False)
            image_button.click(
                detect_image,
                inputs=[image_input, image_conf, image_iou],
                outputs=[image_output, image_summary],
                concurrency_limit=1,
            )

        with gr.Tab("视频检测"):
            with gr.Row():
                video_input = gr.Video(label="上传视频", sources=["upload"])
                video_output = gr.Video(label="检测结果")
            with gr.Row():
                video_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="置信度")
                video_iou = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="IoU")
            video_button = gr.Button("开始检测", variant="primary")
            video_summary = gr.Textbox(label="运行统计", interactive=False)
            video_button.click(
                detect_video,
                inputs=[video_input, video_conf, video_iou],
                outputs=[video_output, video_summary],
                concurrency_limit=1,
            )


if __name__ == "__main__":
    demo.launch()
