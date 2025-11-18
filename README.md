# VectorMind-RAG--AI-Powered-Knowledge-Retrieval-System
Retrieval-Augmented Generation | FAISS Vector DB | Sentence Transformers | Mistral-7B | GPU
VectorMind RAG is an enterprise-grade AI knowledge retrieval system that combines Large Language Models (LLMs) with vector search to deliver factual, context-aware, and trustworthy answers. It solves the biggest issue in AI — hallucinations — by grounding every response on real documents.

## Key Highlights
- High-accuracy embeddings using all-mpnet-base-v2
- FAISS vector database with ~12,474 dense embeddings
- Mistral-7B-Instruct for natural language generation
- Dataset: HuggingFace Documentation
- Cosine similarity–based intelligent retrieval
- Modular: plug-and-play for any domain (medical, legal, finance, enterprise)
- Designed for scalability, extensibility, and production-level RAG workflows

  | Component  | Technology                                      |
| ---------- | ----------------------------------------------- |
| Embeddings | `all-mpnet-base-v2`                             |
| Vector DB  | **FAISS**                                       |
| LLM        | Mistral-7B-Instruct                             |
| Frameworks | Transformers, LangChain (optional), HuggingFace |
| Notebook   | Jupyter / Kaggle                                |
| Language   | Python                                          |

## Why This Project Matters
-Modern LLMs are powerful, but they hallucinate.
-Enter RAG — Retrieval Augmented Generation.
# This project helps users and companies:
-Retrieve accurate information from private datasets
-Build AI assistants for PDFs, FAQs, policies, manuals
-Create domain-specific ChatGPT alternatives
-Reduce hallucination by grounding answers in real knowledge
-Enable faster search & decision-making inside organizations

## Architecture Overview
          ┌──────────────┐
          │ Raw Documents │
          └──────┬───────┘
                 │
         (Chunking + Cleaning)
                 │
          ┌──────▼───────┐
          │ Embedding     │ → all-mpnet-base-v2
          └──────┬────────┘
                 │
          ┌──────▼───────┐
          │ FAISS Vector  │
          │   Database    │
          └──────┬────────┘
                 │
          ┌──────▼────────┐
          │   Retriever    │ (Cosine Similarity)
          └──────┬────────┘
                 │
          ┌──────▼────────┐
          │   Mistral 7B   │ → Final Answer
          └───────────────┘

## Project Structure
 IntelliSearch-RAG
│── rag-project.ipynb
│── rag_metadata.json
│── requirements.txt
└── README.md

## Performance Notes
-Embedding Model: all-mpnet-base-v2
-LLM: Mistral-7B
-Dataset: huggingface_doc
-Chunks Generated: 12,474
-Vector DB: FAISS
-Similarity Metric: Cosine
# These choices ensure:
-High semantic accuracy
-Fast vector retrieval
-Low hallucination rates
-High-quality responses






