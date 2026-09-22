from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://github.com/Leo-0502/sea-sentinel-yolo26/"
    "releases/download/v1.0.0/sea-sentinel-yolo26.pt"
)


def ensure_model(path: Path) -> Path:
    """Download the released model weights when they are not available locally."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        urlretrieve(MODEL_URL, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
