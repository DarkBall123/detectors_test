import subprocess
import sys
from pathlib import Path


def test_flake8():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "src", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"flake8 завершился с ошибкой:\n{result.stdout}\n{result.stderr}"
        )
