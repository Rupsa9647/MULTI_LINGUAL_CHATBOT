import os
import json
from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FAISS_DIR = "data/faiss"
DEFAULT_BATCH_SIZE = 64


# ---------------- HELPER FUNCTIONS ----------------
def load_chunk_texts(chunk_folder: Path):
    """Load text and metadata from all chunk JSON files."""
    chunk_files = list(chunk_folder.glob("*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunk_folder}")

    texts, metadata = [], []
    for f in chunk_files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            texts.append(data["text"])
            metadata.append(data["metadata"])
    return texts, metadata, chunk_files


def train_ivf_index(model, texts, dim, n_list):
    """Train an IVF index on a representative sample of embeddings."""
    print("🧠 Training IVF index...")
    sample_texts = texts[:min(1000, len(texts))]
    sample_embeddings = np.array(model.encode(sample_texts, show_progress_bar=True)).astype("float32")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_list, faiss.METRIC_L2)
    index.train(sample_embeddings)
    print("✅ IVF index trained.")
    return index


# ---------------- MAIN FUNCTION ----------------
def embed_chunks(
    chunk_folder: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    faiss_dir: str = DEFAULT_FAISS_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Dynamically embeds all chunk JSONs in `chunk_folder` using a multilingual model,
    builds a FAISS IVF index, and saves both index & metadata.

    Returns:
        total_chunks (int)
        faiss_index_path (str)
        metadata_path (str)
    """
    chunk_folder = Path(chunk_folder)
    if not chunk_folder.exists():
        raise FileNotFoundError(f"Chunk folder not found: {chunk_folder}")

    os.makedirs(faiss_dir, exist_ok=True)

    # Construct output paths dynamically
    index_name = f"{chunk_folder.stem}_multilang_index_ivf.faiss"
    metadata_name = f"{chunk_folder.stem}_multilang_metadata.json"
    faiss_index_path = Path(faiss_dir) / index_name
    metadata_path = Path(faiss_dir) / metadata_name

    # Load model
    print(f"🚀 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Load chunk texts
    texts, metadata_list, chunk_files = load_chunk_texts(chunk_folder)
    print(f"📄 Found {len(texts)} chunks to embed.")

    # Setup index
    dummy_vec = model.encode(["test"], show_progress_bar=False)
    dim = len(dummy_vec[0])
    n_list = min(1000, len(texts))

    # Train index
    index = train_ivf_index(model, texts, dim, n_list)

    # Stream embeddings
    for i in tqdm(range(0, len(texts), batch_size), desc="🔹 Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(faiss_index_path))
    print(f"✅ FAISS index saved to: {faiss_index_path}")

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata_list, mf, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")

    print(f"\n✅ Successfully embedded {len(texts)} chunks.")
    return len(texts), str(faiss_index_path), str(metadata_path)


# ---------------- PIPELINE COMPATIBLE ENTRY ----------------
if __name__ == "__main__":
    chunk_dir = input("📁 Enter chunk folder path: ").strip()
    try:
        total, index_path, meta_path = embed_chunks(chunk_dir)
        print(f"\n✅ Embedding complete!")
        print(f"Total chunks: {total}")
        print(f"FAISS Index: {index_path}")
        print(f"Metadata: {meta_path}")
    except Exception as e:
        print(f"❌ Error during embedding: {e}")
