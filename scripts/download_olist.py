"""
Download the Olist Brazilian E-Commerce dataset from Kaggle.

Reads KAGGLE_API_TOKEN from .env (JSON string from kaggle.json),
writes ~/.kaggle/kaggle.json, then downloads the dataset.

Usage:
    uv run scripts/download_olist.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Setup kaggle credentials ---
username = os.getenv("KAGGLE_USERNAME")
key      = os.getenv("KAGGLE_KEY")
if not username or not key:
    raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in .env")

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"
kaggle_json.write_text(json.dumps({"username": username, "key": key}))
kaggle_json.chmod(0o600)

print(f"Kaggle credentials set for user: {username}")

# --- Download dataset ---
import kaggle

kaggle.api.authenticate()

out_dir = Path("data/olist")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading olistbr/brazilian-ecommerce (~45MB)...")
kaggle.api.dataset_download_files(
    "olistbr/brazilian-ecommerce",
    path=str(out_dir),
    unzip=True,
    quiet=False,
)

files = list(out_dir.glob("*.csv"))
print(f"\nDone. {len(files)} CSV files saved to {out_dir}/")
for f in sorted(files):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:55s} {size_kb:>6} KB")
