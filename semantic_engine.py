# semantic_engine

import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util

logging.info("🔍 Загрузка модели семантики (SBERT)...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]):
    return sbert_model.encode(texts, convert_to_tensor=True)

def semantic_search(query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    logging.info(f"🔍 Семантический поиск для: {query}")
    query_emb = embed_texts([query])
    cand_emb = embed_texts(candidates)
    hits = util.semantic_search(query_emb, cand_emb, top_k=top_k)[0]
    return [(candidates[hit['corpus_id']], hit['score']) for hit in hits]

def select_samples_by_semantics(prompt: str, sample_index: Dict[str, Dict]) -> List[str]:
    candidates = list(sample_index.keys())
    ranked = semantic_search(prompt, candidates, top_k=10)
    return [name for name, score in ranked]
