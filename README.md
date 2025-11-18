# VectorIQ-RAG--AI-Powered-Knowledge-Retrieval-System
VectorIQ RAG is an enterprise-grade AI knowledge retrieval system that combines Large Language Models (LLMs) with vector search to deliver factual, context-aware, and trustworthy answers. It solves the biggest issue in AI — hallucinations — by grounding every response on real documents.

## Key Highlights
- High-accuracy embeddings using all-mpnet-base-v2
- FAISS vector database with ~12,474 dense embeddings
- Mistral-7B-Instruct for natural language generation
- Dataset: HuggingFace Documentation
- Cosine similarity–based intelligent retrieval
- Modular: plug-and-play for any domain (medical, legal, finance, enterprise)
- Designed for scalability, extensibility, and production-level RAG workflows

## Why This Project Matters
Modern LLMs are powerful, but they hallucinate.
## This project helps users and companies:
- Retrieve accurate information from private datasets
- Build AI assistants for PDFs, FAQs, policies, manuals
- Create domain-specific ChatGPT alternatives
- Reduce hallucination by grounding answers in real knowledge
- Enable faster search & decision-making inside organizations

## Architecture Overview
Documents → Chunking → Embeddings → FAISS DB → Retriever → Mistral-7B → Final Answer

## Project Metadata
- Embedding Model: all-mpnet-base-v2
- Language Model: Mistral-7B-Instruct
- Vector Database: FAISS
- Similarity Metric: Cosine
- Chunks Created: 12,474
- Dataset: HuggingFace documentation
These selections provide strong semantic retrieval performance with efficient storage.

## Tech Stack
- Python
- FAISS
- Sentence Transformers
- HuggingFace Transformers
- tqdm
- Kaggle-GPU
- 
## These choices ensure:
-High semantic accuracy
-Fast vector retrieval
-Low hallucination rates
-High-quality responses






