import os
import subprocess

def test_build_pipeline_runs():
    # Just check the script completes without error (artifacts are created)
    cmd = ["python","-m","src.pipeline","--mode","build"]
    subprocess.check_call(cmd)
    assert os.path.exists("artifacts/courses.parquet")
