# src/utils.py
import os

from pathlib import Path


def ensure_dirs():
 Path("datas/original_pdf").mkdir(parents=True, exist_ok=True)
 Path("datas/chunks").mkdir(parents=True, exist_ok=True)
 Path("datas/embeddings").mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":
 ensure_dirs()