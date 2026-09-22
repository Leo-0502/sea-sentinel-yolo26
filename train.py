import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model for vessel detection")
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained model or checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, mps")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="sea_sentinel")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")
    YOLO(args.model).train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=args.project, name=args.name,
    )


if __name__ == "__main__":
    main()
