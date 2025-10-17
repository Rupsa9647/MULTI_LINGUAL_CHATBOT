import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional


# ---------------- DEFAULT CONFIG ----------------
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 3


class Reranker:
    """
    Multilingual reranker using a CrossEncoder model.
    Works with outputs from the HybridRetriever.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize a multilingual cross-encoder reranker.
        Automatically uses GPU if available.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading reranker model ({model_name}) on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("✅ Reranker ready.")

    # ---------------- MAIN FUNCTION ----------------
    def rerank_results(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved chunks using a multilingual cross-encoder.

        Args:
            query (str): The user query
            retrieved_results (list): Results from HybridRetriever.hybrid_search()
            top_k (int): Number of top results to return after reranking

        Returns:
            List[dict]: Top reranked results, sorted by rerank score
        """
        if not retrieved_results:
            print("⚠️ No retrieved results to rerank.")
            return []

        # Create (query, document_text) pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in retrieved_results]

        print(f"🔹 Computing rerank scores for {len(pairs)} candidates...")
        scores = self.model.predict(pairs)

        # Attach new scores
        for i, doc in enumerate(retrieved_results):
            doc["rerank_score"] = float(scores[i])

        # Sort and return top_k
        reranked = sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)
        print(f"✅ Reranked top {top_k} results.")
        return reranked[:top_k]


# ---------------- PIPELINE-COMPATIBLE ENTRY ----------------
if __name__ == "__main__":
    from retriever import HybridRetriever  # <-- your modular retriever

    print("📄 Multilingual Reranker Module")
    faiss_path = input("Enter FAISS index path: ").strip()
    metadata_path = input("Enter metadata path: ").strip()
    chunk_folder = input("Enter chunk folder path: ").strip()
    query = input("🔍 Enter your query: ").strip()

    # Step 1: Initialize retriever
    retriever = HybridRetriever(
        faiss_index_path=faiss_path,
        metadata_path=metadata_path,
        chunk_folder=chunk_folder,
    )

    # Step 2: Retrieve top 20 candidates
    print("\n🔹 Retrieving top 20 candidates...")
    retrieved_results = retriever.hybrid_search(query, top_k=20)
    print(f"✅ Retrieved {len(retrieved_results)} candidates. Now reranking...")

    # Step 3: Initialize and rerank
    reranker = Reranker()
    final_results = reranker.rerank_results(query, retrieved_results, top_k=5)

    # Step 4: Display final reranked output
    print("\n🔝 Final Reranked Results:")
    for i, res in enumerate(final_results, 1):
        print(f"\n{i}. [Score: {res['rerank_score']:.4f}]")
        print(f"   Language: {res.get('language', 'unknown')}")
        print(f"   Source: {res.get('source_file', 'unknown')}")
        print(f"   Text Preview: {res['text'][:500]}...")
