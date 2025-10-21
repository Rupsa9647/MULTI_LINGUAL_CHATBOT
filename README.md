# MULTI_LINGUAL_CHATBOT# 📄 Multilingual RAG Chatbot

**Version:** 1.0  
**Last Updated:** October 2025  
**Author:** Rupsa Jana

---

## Overview

The **Multilingual RAG Chatbot** is an AI system that allows users to upload PDF documents and interactively chat with the content. It uses a **retrieval-augmented generation (RAG)** approach combining semantic search, keyword search, and LLM-based generation. It supports multilingual content and maintains session-based chat history.

---

## Features

- Upload both scanned or digital PDFs and process them into chunks.
- Embedding of chunks using Sentence Transformers.
- FAISS-based semantic search.
- BM25 keyword-based search.
- Cross-encoder reranking for top retrieved chunks.
- Memory-aware multilingual LLM response generation (Gemini 2.5 Flash).
- Persistent chat history using SQLite and JSON.
- Streamlit interface for interactive chat.

---

## Requirements

- Python 3.10+
- Streamlit
- FAISS (CPU version)
- Sentence Transformers
- LangChain
- torch
- rank-bm25
- python-dotenv
- SQLite3

Install dependencies:

```bash
pip install -r requirements.txt


## API KEY
GOOGLE_API_KEY=your_google_gemini_api_key_here


## Usage

Upload PDFs in the sidebar and click Process PDF(s).

Ask questions in the chat input box.

The AI will retrieve relevant chunks, rerank them, and generate responses.

Previous sessions are displayed in the sidebar.




## Maintenance

Clean old data: Delete data/chunks, data/faiss_index, data/metadata, chat_history.db.

Update models: Change embedding or reranker model paths in retriever.py and reranker.py.

API Key rotation: Update .env with a new GOOGLE_API_KEY.

Backup chat: Export chat_history.db or chat_history.json.


## Troubleshooting

GOOGLE_API_KEY not found → Ensure .env exists and key is correct.

No relevant context found → PDF chunking or indexing may have failed.

Retriever initialization errors → Check FAISS index files and metadata.