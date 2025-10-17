
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import unicodedata
from pathlib import Path

class Retriever:
    def __init__(
        self,
        chunk_folder: str,
        faiss_index_path: str,
        metadata_path: str,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        top_k: int = 10
    ):
        self.chunk_folder = Path(chunk_folder)
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

        # Load model
        self.model = SentenceTransformer(self.embedding_model_name)

        # Load FAISS index
        self.index = faiss.read_index(self.faiss_index_path)

        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load texts for BM25
        chunk_files = list(self.chunk_folder.glob("*.json"))
        self.texts = []
        for f in chunk_files:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                text = data.get("text", "").strip()
                if text:
                    self.texts.append(text)

        tokenized_corpus = [self._clean_and_tokenize(t) for t in self.texts if len(self._clean_and_tokenize(t)) > 0]
        if not tokenized_corpus:
            raise ValueError("No valid text found to build BM25.")
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _clean_and_tokenize(self, text):
        text = unicodedata.normalize("NFKC", text)
        tokens = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
        return tokens

    def hybrid_search(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k

        # Semantic search
        query_vec = np.array(self.model.encode([query])).astype("float32")
        semantic_dists, semantic_indices = self.index.search(query_vec, top_k)
        semantic_scores = {}
        for dist, idx in zip(semantic_dists[0], semantic_indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            semantic_scores[idx] = 1 / (1 + dist)

        # Keyword search
        tokenized_query = self._clean_and_tokenize(query)
        keyword_scores = self.bm25.get_scores(tokenized_query)
        top_keyword_indices = np.argsort(keyword_scores)[::-1][:top_k]
        keyword_scores_dict = {int(i): float(keyword_scores[i]) for i in top_keyword_indices}

        # Combine
        combined_scores = {}
        all_indices = set(semantic_scores.keys()) | set(keyword_scores_dict.keys())
        for idx in all_indices:
            sem = semantic_scores.get(idx, 0)
            key = keyword_scores_dict.get(idx, 0)
            combined = self.semantic_weight * sem + self.keyword_weight * key
            combined_scores[idx] = combined

        # Sort
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (idx, score) in enumerate(sorted_results, start=1):
            m = self.metadata[idx] if idx < len(self.metadata) else {}
            text = self.texts[idx] if idx < len(self.texts) else ""
            results.append({
                "rank": rank,
                "score": round(float(score), 4),
                "language": m.get("language_hint", "unknown"),
                "source_file": m.get("source_file", "unknown"),
                "text": text[:500]
            })
        return results
