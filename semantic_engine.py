# semantic_engine

import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
    logging.info("🔍 Загрузка модели семантики (SBERT)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError as e:
    SBERT_AVAILABLE = False
    logging.warning(f"⚠️ SBERT не доступен: {e}")
    sbert_model = None

def embed_texts(texts: List[str]):
    """Эмбеддинг текстов"""
    if not SBERT_AVAILABLE or sbert_model is None:
        logging.warning("⚠️ SBERT не доступен, используем простой поиск")
        return None
    
    try:
        return sbert_model.encode(texts, convert_to_tensor=True)
    except Exception as e:
        logging.error(f"❌ Ошибка эмбеддинга: {e}")
        return None

def semantic_search(query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Семантический поиск"""
    if not SBERT_AVAILABLE or sbert_model is None:
        # Fallback: простой поиск по подстрокам
        logging.warning("⚠️ Используем простой поиск вместо семантического")
        matches = []
        query_lower = query.lower()
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            # Простая оценка по совпадению слов
            query_words = set(query_lower.split())
            candidate_words = set(candidate_lower.split())
            score = len(query_words.intersection(candidate_words)) / max(len(query_words), 1)
            
            if score > 0 or query_lower in candidate_lower:
                matches.append((candidate, score))
        
        # Сортируем по скору и возвращаем top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    try:
        logging.info(f"🔍 Семантический поиск для: {query}")
        query_emb = embed_texts([query])
        if query_emb is None:
            return []
        
        cand_emb = embed_texts(candidates)
        if cand_emb is None:
            return []
        
        hits = util.semantic_search(query_emb, cand_emb, top_k=top_k)[0]
        return [(candidates[hit['corpus_id']], hit['score']) for hit in hits]
    
    except Exception as e:
        logging.error(f"❌ Ошибка семантического поиска: {e}")
        return []

def select_samples_by_semantics(prompt: str, sample_index: Dict[str, Dict], max_results: int = 10) -> List[str]:
    """Выбор сэмплов по семантике"""
    if not sample_index:
        logging.warning("⚠️ Пустой индекс сэмплов")
        return []
    
    try:
        candidates = list(sample_index.keys())
        if not candidates:
            return []
        
        ranked = semantic_search(prompt, candidates, top_k=max_results)
        result = [name for name, score in ranked if score > 0.1]  # Минимальный порог
        
        logging.info(f"🔍 Найдено семантических совпадений: {len(result)}")
        return result[:max_results]
    
    except Exception as e:
        logging.error(f"❌ Ошибка выбора сэмплов: {e}")
        return []
