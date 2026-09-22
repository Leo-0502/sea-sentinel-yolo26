---
title: Sea Sentinel YOLO26
emoji: "🌊"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 海面哨兵：基于 YOLO26 的船只实时监测与告警系统

面向近海、港口和重点水域视频监控的桌面端原型。系统使用 Ultralytics YOLO 完成船只检测，结合 ByteTrack 保持跨帧目标 ID，并在船只进入监测画面时触发声光告警、记录事件和保存取证截图。

## 功能

- 支持图片、视频文件和摄像头实时输入
- 自动加载自训练权重；首次克隆若本地缺失，将从 GitHub Release 下载，也可导入其他 `.pt` / `.onnx` 模型
- 自动识别单类自训练模型；对 COCO 模型自动筛选 `boat` 类，避免硬编码类别导致漏检
- 基于 ByteTrack 的多目标跟踪与可回收显示 ID
- 置信度、IoU 阈值可调，实时展示目标数、最高置信度和推理 FPS
- 船只进入/离开状态驱动的声音及界面告警
- CSV 事件日志和一键检测截图

## 系统流程

```text
图片 / 视频 / 摄像头
        |
        v
YOLO26 检测 -> ByteTrack 跟踪 -> ID 映射 -> 检测框与实时指标
                                      |
                                      v
                          进入/离开状态机
                              |        |
                              v        v
                           声光告警   CSV 日志/截图
```

## 环境与运行

推荐 Python 3.10-3.12。首次运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
python app_main.py
```

应用启动时会自动加载 `海面船只检测.pt`；文件缺失时会从项目的 `v1.0.0` Release 自动下载。点击“打开媒体”可直接使用 `test_video/生成船只.mp4` 样例视频，然后点击“开始监测”。检测事件写入 `outputs/events.csv`，截图写入 `outputs/snapshots/`。较大的 `boat.mp4` 不纳入 Git 仓库，避免仓库体积膨胀。

### Gradio Web 版

用于 Hugging Face Spaces 或本地浏览器演示：

```powershell
python app.py
```

默认访问地址为 `http://127.0.0.1:7860`。Web 版支持图片和视频检测、置信度/IoU 调节、ByteTrack 跟踪、结果预览与下载。Spaces 会读取本文件顶部的配置并自动运行 `app.py`。

## 训练

数据集应采用 Ultralytics YOLO 格式，并提供 `data.yaml`：

```powershell
python train.py --data path\to\data.yaml --model yolo26n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

所有参数均可通过 `python train.py --help` 查看。训练结果默认写入 `runs/train/sea_sentinel`。

## 测试

核心逻辑测试不依赖 GPU 或模型权重：

```powershell
python -m unittest discover -s tests -v
```

## 项目结构

```text
app_main.py                 # PySide6 桌面端、视频线程和推理流程
app.py                      # Hugging Face Gradio Web 入口
sea_sentinel/core.py        # ID 管理、类别解析与事件日志
train.py                    # 参数化训练入口
tests/test_core.py          # 核心逻辑单元测试
test_video/                 # 演示视频
海面船只检测.pt              # 自训练权重
警告.mp3 / 警报音效.mp3      # 告警声音
简历项目描述.md              # 与现有简历风格一致的项目文案
```

## 已知边界

- 实际精度与速度取决于训练数据、设备和输入分辨率；仓库不声明未经验证的 mAP 或 FPS。
- 当前告警条件为画面中出现任意目标，尚未加入电子围栏、目标身份白名单或跨摄像头 ReID。
- 首次使用通用预训练权重时，Ultralytics 可能需要联网下载；本项目附带的权重无需下载。
