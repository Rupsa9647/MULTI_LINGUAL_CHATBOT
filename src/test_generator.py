from pathlib import Path
from chunking import process_pdf
from embed_and_index import embed_chunks
from retriever import Retriever
from generator import Generator

# -------------------------
# 1️⃣ Process PDF into chunks
# -------------------------
pdf_file = r"D:\multi_language_rag\Data\bn\15092024_142.pdf"  # Your PDF path
chunk_output_dir = Path("data/chunks") / Path(pdf_file).stem

print("🔹 Processing PDF into chunks...")
try:
    chunk_files, chunk_dir = process_pdf(pdf_file, output_base_dir=chunk_output_dir)
    print(f"✅ PDF processed into {len(chunk_files)} chunks at {chunk_dir}")
except FileNotFoundError:
    print(f"❌ PDF not found: {pdf_file}")
    exit(1)

# -------------------------
# 2️⃣ Embed chunks + build FAISS index
# -------------------------
print("🔹 Embedding chunks and building FAISS index...")
try:
    total_chunks, faiss_index_path, metadata_path = embed_chunks(chunk_dir)
    print(f"✅ Embedded {total_chunks} chunks")
    print(f"FAISS index saved at: {faiss_index_path}")
    print(f"Metadata saved at: {metadata_path}")
except Exception as e:
    print(f"❌ Error during embedding: {e}")
    exit(1)

# -------------------------
# 3️⃣ Initialize Retriever
# -------------------------
retriever = Retriever(
    chunk_folder=chunk_dir,
    faiss_index_path=faiss_index_path,
    metadata_path=metadata_path,
    semantic_weight=0.6,
    keyword_weight=0.4,
    top_k=10
)
print("✅ Retriever ready")

# -------------------------
# 4️⃣ Initialize Generator (includes optional Reranker)
# -------------------------
try:
    generator = Generator(use_reranker=True)
    print("✅ Generator ready")
except Exception as e:
    print(f"❌ Error initializing Generator: {e}")
    exit(1)

# -------------------------
# 5️⃣ Run query through the pipeline
# -------------------------
user_query = "Explain the main idea of the document in simple terms."

# Retrieve top chunks
retriever_results = retriever.hybrid_search(user_query)
if not retriever_results:
    print("⚠ No chunks retrieved. Exiting.")
    exit(0)

# Generate answer
answer = generator.generate_answer(user_query, retriever_results)

# -------------------------
# 6️⃣ Display results
# -------------------------
print("\n--- Query Result ---")
print(f"User Query: {user_query}")
print(f"Answer:\n{answer}")
