"""Shared pytest fixtures.

Builds a tiny TF-IDF artifact set in a temp dir so tests exercise the real
pipeline + recommender + API without heavy models or network access.
"""

import pandas as pd
import pytest
import yaml

SAMPLE_ROWS = [
    ("Deep Learning with PyTorch", "DeepLearning.AI", "pytorch, neural networks, deep learning", "Course", "beginner", 4.7),
    ("Machine Learning with Python", "Coursera", "python, scikit-learn, machine learning", "Course", "beginner", 4.8),
    ("Advanced TensorFlow", "Google", "tensorflow, deep learning, keras", "Course", "advanced", 4.5),
    ("SQL for Data Analysis", "IBM", "sql, databases, analytics", "Course", "beginner", 4.6),
    ("Cloud Computing on AWS", "Amazon", "aws, cloud, ec2, s3", "Specialization", "intermediate", 4.4),
    ("Natural Language Processing", "Stanford", "nlp, transformers, python, deep learning", "Course", "advanced", 4.9),
    ("Data Visualization with Python", "Coursera", "python, matplotlib, plotly", "Course", "beginner", 4.3),
    ("Reinforcement Learning", "DeepMind", "rl, python, deep learning, pytorch", "Course", "advanced", 4.7),
]


@pytest.fixture()
def built_artifacts(tmp_path, monkeypatch):
    """Build TF-IDF artifacts into a temp artifact dir; yield paths + config."""
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    art_dir.mkdir()

    df = pd.DataFrame(SAMPLE_ROWS, columns=["course", "partner", "skills", "certificatetype", "level", "rating"])
    csv_path = data_dir / "sample.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "columns": {
            "id": None, "title": "course", "description": "partner",
            "skills": "skills", "category": "certificatetype",
            "level": "level", "rating": "rating", "url": None,
        },
        "text_fields": ["title", "title", "skills", "skills", "description", "level", "category"],
        "min_characters": 5,
        "use_sbert": False,
        "kmeans_k": 3,
        "top_k": 5,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("ARTIFACT_STORE", "local")
    monkeypatch.setenv("ARTIFACT_DIR", str(art_dir))
    # Keep the suite hermetic: never reach a real LLM provider, even when a
    # developer's .env configures one. Process env wins over .env.
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    monkeypatch.setenv("LLM_ENABLED", "false")

    from src.pipeline import build
    from src.storage import get_artifact_store

    build(str(cfg_path), str(csv_path), store=get_artifact_store())

    return {"art_dir": art_dir, "cfg_path": str(cfg_path)}
