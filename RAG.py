from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

dataset = load_dataset("m-ric/huggingface_doc", split="train")

print(f"Dataset size: {len(dataset)} documents")
print(f"Sample document:\n{dataset[0]}")
df = pd.DataFrame({
    'text': [doc['text'] for doc in dataset],
    'source': [doc['source'] for doc in dataset]
})
print(f"\nDataFrame shape: {df.shape}")
print(f"First few sources:\n{df['source'].unique()[:5]}")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

df_clean = df.dropna(subset=['text'])
print(f"Documents after cleaning: {len(df_clean)}")

documents = [
    LangchainDocument(page_content=row['text'], metadata={"source": row['source']})
    for _, row in df_clean.iterrows()
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          
    chunk_overlap=100,         
    separators=["\n\n", "\n", ".", " "]
)

docs_processed = []
for doc in tqdm(documents[:500], desc="Chunking documents"):  
    chunks = text_splitter.split_documents([doc])
    docs_processed.extend(chunks)
print(f"Total chunks created: {len(docs_processed)}")
print(f"Sample chunk:\n{docs_processed[0].page_content[:200]}...")
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL = "all-mpnet-base-v2"

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Creating embeddings for all chunks...")
chunk_texts = [doc.page_content for doc in docs_processed]
embeddings = embedding_model.encode(
    chunk_texts,
    batch_size=4,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
import faiss
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


hf_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},  
    encode_kwargs={"normalize_embeddings": True}  
)

print("Building FAISS vector database")
vector_db = FAISS.from_documents(
    docs_processed,
    hf_embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

print(" FAISS database created")

vector_db.save_local("faiss_index")
print(" Database saved locally")
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

print(f"Loading LLM: {LLM_MODEL}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

llm_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL,
    device=1 if device == "cuda" else -1,
    torch_dtype=torch.float16,
    max_new_tokens=128,
    temperature=0.3,
    num_beams=1, 
    top_p=0.95
)

print(" LLM loaded successfully")
RAG_PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer concisely based only on the above context. If the answer is not found, respond: "I don't know."
"""
from typing import Tuple, List

def rag_retrieve_and_generate(
    question: str,
    vector_db,
    llm_pipeline,
    k: int = 5
) -> Tuple[str, List]:
    
    print(f" Retrieving documents for: {question}")
    retrieved_docs = vector_db.similarity_search(question, k=k)
    
    context = "\n".join([
        f"Source {i+1}: {doc.page_content[:300]}...\n(Source: {doc.metadata.get('source', 'Unknown')})\n"
        for i, doc in enumerate(retrieved_docs)
    ])
    

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    print(" Generating answer...")
    response = llm_pipeline(prompt, max_new_tokens=256)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    return answer, retrieved_docs
test_question = "What are transformers in machine learning?"
answer, sources = rag_retrieve_and_generate(
    test_question,
    vector_db,
    llm_pipeline,
    k=5
)

print(f"\n{'='*60}")
print(f"Question: {test_question}")
print(f"{'='*60}")
print(f"Answer:\n{answer}")
print(f"{'='*60}")
print(f"Retrieved Sources:")
for i, doc in enumerate(sources):
    print(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
    print(f"   Content preview: {doc.page_content[:150]}...")
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rag_with_reranking(
    question: str,
    vector_db,
    llm_pipeline,
    reranker,
    k_retrieve: int = 15,
    k_final: int = 5
) -> Tuple[str, List]:
    
    retrieved_docs = vector_db.similarity_search(question, k=k_retrieve)
    
    doc_text_pairs = [(question, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(doc_text_pairs)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_final]
    reranked_docs = [retrieved_docs[i] for i in ranked_indices]
    
    context = "\n".join([
        f"Document {i+1}: {doc.page_content[:400]}...\n"
        for i, doc in enumerate(reranked_docs)
    ])
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm_pipeline(prompt)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    
    return answer, reranked_docs

answer_reranked, sources_reranked = rag_with_reranking(
    "What is transfer learning in neural networks?",
    vector_db,
    llm_pipeline,
    reranker
)

print(f"Answer (with reranking):\n{answer_reranked}")
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_rag_quality(question: str, answer: str, retrieved_docs, embeddings_model):

    qa_embedding = embeddings_model.encode(f"{question} {answer}")
    doc_embeddings = embeddings_model.encode([doc.page_content for doc in retrieved_docs])
    
    qa_relevance = cosine_similarity([qa_embedding], doc_embeddings).mean()
    
    question_embedding = embeddings_model.encode(question)
    doc_relevance = cosine_similarity([question_embedding], doc_embeddings).mean()
    
    return {
        "qa_relevance": round(qa_relevance, 4),
        "doc_relevance": round(doc_relevance, 4),
        "retrieval_count": len(retrieved_docs)
    }

metrics = evaluate_rag_quality(
    "What are transformers?",
    answer,
    sources,
    embedding_model
)
print(f"\n RAG Quality Metrics:")
print(f"  QA Relevance Score: {metrics['qa_relevance']} (higher is better)")
print(f"  Document Relevance: {metrics['doc_relevance']} (higher is better)")
print(f"  Documents Retrieved: {metrics['retrieval_count']}")
def interactive_rag_demo():
    """
    Interactive demo for testing RAG system
    """
    print("\n" + "="*60)
    print(" AI/ML Documentation RAG System")
    print("="*60)
    
    test_questions = [
        "How do I use transformers for NLP tasks?",
        "What is transfer learning?",
        "Explain attention mechanisms in detail",
        "How do I fine-tune a pretrained model?",
        "What are embeddings and why are they useful?"
    ]
    
    for question in test_questions:
        print(f"\n Question: {question}")
        print("-" * 60)
        
        answer, retrieved_docs = rag_retrieve_and_generate(
            question,
            vector_db,
            llm_pipeline,
            k=3
        )
        
        metrics = evaluate_rag_quality(question, answer, retrieved_docs, embedding_model)
        
        print(f" Answer: {answer[:300]}...")
        print(f" Relevance Score: {metrics['doc_relevance']}")
        print()
interactive_rag_demo()
import json

rag_metadata = {
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "dataset": "m-ric/huggingface_doc",
    "chunks_created": len(docs_processed),
    "vector_db_type": "FAISS",
    "distance_metric": "cosine"
}

with open("rag_metadata.json", "w") as f:
    json.dump(rag_metadata, f, indent=2)

print(" RAG system ready for deployment!")
print(f" Saved files: faiss_index/, rag_metadata.json")

    
