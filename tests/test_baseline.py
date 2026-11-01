import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_command_trains_and_exports_features(tmp_path):
    data = ROOT.parent / "hw0/data"
    if not (data / "train-images-idx3-ubyte.gz").is_file():
        import pytest
        pytest.skip("MNIST files are optional local smoke-test data")
    output = tmp_path / "baseline.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "apps/train_baseline.py"), "--data-dir", str(data),
         "--train-limit", "128", "--eval-limit", "64", "--epochs", "1",
         "--batch-size", "32", "--hidden-dim", "16", "--feature-dim", "8",
         "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record["dataset"] == "MNIST"
    assert record["feature_shape"] == [64, 8]
    assert len(record["epochs"]) == 1
    assert 0 <= record["epochs"][0]["validation_accuracy"] <= 1
    assert record["epochs"][0]["training_seconds"] > 0
