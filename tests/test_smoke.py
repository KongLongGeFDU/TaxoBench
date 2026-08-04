import json
import subprocess
import sys


def test_cli_smoke():
    result = subprocess.run([sys.executable, "-m", "taxobench.score", "--data", "examples/toy_reference.jsonl", "--predictions", "examples/toy_prediction.jsonl"], check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    assert summary["n_scored"] == 1
    assert summary["leaf_ari"] == 1.0
