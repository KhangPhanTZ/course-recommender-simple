import os
import json
import pickle
import pandas as pd
from scipy import sparse
from typing import Any

ART_DIR = "artifacts"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_pickle(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_sparse_matrix(mat, path: str):
    ensure_dir(os.path.dirname(path))
    sparse.save_npz(path, mat)

def load_sparse_matrix(path: str):
    return sparse.load_npz(path)

def save_parquet(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)

def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
