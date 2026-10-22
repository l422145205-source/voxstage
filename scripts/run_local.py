from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'runtime/deps/common')]
from voxstage.cli import main
main()
