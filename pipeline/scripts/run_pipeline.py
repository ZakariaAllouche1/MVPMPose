from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from configs import local as cfg
from src.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(cfg)