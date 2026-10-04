import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime"))
os.environ.setdefault("WANDB_MODE", "disabled")
