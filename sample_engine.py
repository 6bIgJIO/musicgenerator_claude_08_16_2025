# sample_engine.py - Семантический движок подбора сэмплов
import re
import os
import json
import logging
import numpy as np
import asyncio
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

# ML и семантический анализ
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("Semantic analysis libraries not available")

# Аудио анализ
import librosa
import soundfile as sf
from pydub import AudioSegment, effects
from pydub.effects import compress_dynamic_range, normalize

from config import config

# Константы для жанровой классификации
GENRE_KEYWORDS = {
    "trap": ["trap", "drill", "phonk"],
    "lofi": ["lofi", "chill", "jazz", "vintage"],
    "house": ["house", "tech", "deep"],
    "ambient": ["ambient", "pad", "atmosphere"],
    "hip_hop": ["hip", "hop", "boom", "bap"],
    "dnb": ["dnb", "drum", "bass", "jungle"],
    "dubstep": ["dubstep", "wobble", "drop"],
    "techno": ["techno", "industrial", "acid"]
}


@dataclass
class SampleMetadata:
    """Метаданные сэмпла с семантическими характеристиками"""
    # Основные метаданные
    path: str
    filename: str
    duration: float
    tempo: int
    key: Optional[str]
    tags: List[str]
    genres: List[str]
    instrument_role: Optional[str]

    # Аудио характеристики
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc_features: np.ndarray
    chroma_features: np.ndarray

    # Опциональные поля (с default'ами)
    relative_path: Optional[str] = None
    category: Optional[str] = None
    semantic_embedding: Optional[np.ndarray] = None
    semantic_cluster: Optional[int] = None
    similarity_tags: Optional[List[str]] = None
    quality_score: float = 0.0
    energy_level: float = 0.0
    brightness: float = 0.0
    rhythmic_complexity: float = 0.0


class EnhancedSamplePicker:
    def __init__(self, sample_dir, index_file="enhanced_sample_index.json"):
        self.sample_dir = sample_dir
        self.index_file = os.path.join(sample_dir, index_file)

    def analyze_filename_advanced(self, filename):
        name = filename.lower()
        tags = []
        bmp = 120  # Исправлена опечатка: bmp -> bpm
        key = None

        # Паттерны для BPM
        bpm_patterns = [r'(\d{2,3})bpm', r'(\d{2,3})_bpm', r'bmp(\d{2,3})']
        for pattern in bpm_patterns:
            match = re.search(pattern, name)
            if match:
                bmp = int(match.group(1))
                break

        # Паттерны для key
        key_patterns = [r'([a-g][#b]?)[_-]?maj', r'([a-g][#b]?)[_-]?min']
        for pattern in key_patterns:
            match = re.search(pattern, name)
            if match:
                key = match.group(1).upper()
                break

        return tags, bmp, key


class SemanticSampleEngine:
    """
    Продвинутый семантический движок для подбора сэмплов
    
    Возможности:
    - Семантический анализ через SBERT
    - Кластеризация похожих сэмплов  
    - MFCC/хрома анализ для аудио-подобия
    - Жанровая классификация
    - Энергетический анализ
    - Качественная фильтрация
    """
    
    def __init__(self, sample_dir: str = None):
        self.sample_dir = sample_dir or getattr(config, 'DEFAULT_SAMPLE_DIR', './samples')
        self.logger = logging.getLogger(__name__)
        self.enhanced_picker = EnhancedSamplePicker(self.sample_dir)
        
        # Инициализация семантической модели
        self.semantic_model = None
        self.embeddings_cache = {}
        self.sample_clusters = {}
        
        if SEMANTIC_AVAILABLE:
            self._init_semantic_model()
        
        # Загружаем или строим индекс
        self.samples_index: List[SampleMetadata] = []
        self.load_or_build_index()
        
        # Статистика производительности
        self.performance_stats = {
            "queries": 0,
            "cache_hits": 0,
            "avg_query_time": 0.0
        }

    def _init_semantic_model(self):
        """Инициализация семантической модели"""
        try:
            model_name = getattr(config, 'SAMPLE_MATCHING', {}).get("embedding_model", 
                                                  "sentence-transformers/all-MiniLM-L6-v2")
            self.semantic_model = SentenceTransformer(model_name)
            self.logger.info(f"✅ Семантическая модель загружена: {model_name}")

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки семантической модели: {e}")
            self.semantic_model = None

    def analyze_audio_content(self, full_path: str) -> Dict:
        """Глубокий анализ аудиоконтента файла"""
        try:
            # Загружаем аудио через librosa
            max_duration = getattr(config, 'AUDIO_ANALYSIS', {}).get('max_duration', 30)
            sample_rate = getattr(config, 'AUDIO_ANALYSIS', {}).get('sample_rate', 22050)
            
            y, sr = librosa.load(full_path, duration=max_duration, sr=sample_rate)
            
            analysis = {
                "tempo": 120,
                "key": None,
                "content_tags": [],
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0,
                "brightness": 0.0,
                "energy": 0.5,
                "rhythmic_complexity": 0.0,
                "mfcc": np.array([]),
                "chroma": np.array([])
            }
            
            if len(y) > 0:
                # Темп
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                analysis["tempo"] = float(tempo)
                
                # Спектральные характеристики
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                analysis["spectral_centroid"] = float(spectral_centroid.mean())
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                analysis["spectral_rolloff"] = float(spectral_rolloff.mean())
                
                zcr = librosa.feature.zero_crossing_rate(y)
                analysis["zero_crossing_rate"] = float(zcr.mean())
                
                analysis["brightness"] = analysis["spectral_centroid"] / sr
                
                # MFCC
                n_mfcc = getattr(config, 'AUDIO_ANALYSIS', {}).get('n_mfcc', 13)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                analysis["mfcc"] = mfcc.mean(axis=1)
                
                # Хрома
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                analysis["chroma"] = chroma.mean(axis=1)
                
                # Определение тональности через хрома
                if chroma.size > 0:
                    chroma_mean = chroma.mean(axis=1)
                    key_idx = chroma_mean.argmax()
                    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    analysis["key"] = keys[key_idx]
                
                # Энергия
                rms = librosa.feature.rms(y=y)
                analysis["energy"] = float(min(1.0, rms.mean() * 10))
                
                # Ритмическая сложность
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                analysis["rhythmic_complexity"] = len(onset_frames) / (len(y) / sr)
                
                # Контентные теги на основе аудиоанализа
                if analysis["energy"] > 0.7:
                    analysis["content_tags"].append("energetic")
                elif analysis["energy"] < 0.3:
                    analysis["content_tags"].append("calm")
                    
                if analysis["brightness"] > 0.3:
                    analysis["content_tags"].append("bright")
                elif analysis["brightness"] < 0.1:
                    analysis["content_tags"].append("dark")
                    
                if analysis["rhythmic_complexity"] > 5:
                    analysis["content_tags"].append("complex")
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа аудио {full_path}: {e}")
            return {
                "tempo": 120,
                "key": None,
                "content_tags": [],
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0,
                "brightness": 0.0,
                "energy": 0.5,
                "rhythmic_complexity": 0.0,
                "mfcc": np.array([]),
                "chroma": np.array([])
            }

    def build_enhanced_index(self):
        """Построение индекса сэмплов с жанровой привязкой"""
        logging.info("🔍 Начинаю расширенную индексацию сэмплов...")
        enhanced_index = []
        processed = 0
        
        for root, _, files in os.walk(self.sample_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.aiff', '.flac')):
                    full_path = os.path.join(root, file)
                    try:
                        # Базовая информация
                        try:
                            audio = AudioSegment.from_file(full_path)
                            duration = len(audio) / 1000
                        except:
                            # Fallback через librosa
                            y, sr = librosa.load(full_path, duration=30)
                            duration = len(y) / sr
                        
                        # Фильтрация коротких и длинных файлов
                        min_duration = getattr(config, 'QUALITY_FILTERS', {}).get('min_duration', 0.5)
                        max_duration = getattr(config, 'QUALITY_FILTERS', {}).get('max_duration', 300)
                        
                        if duration < min_duration or duration > max_duration:
                            continue
                        
                        # Анализ имени файла
                        filename_tags, filename_bmp, filename_key = self.enhanced_picker.analyze_filename_advanced(file)
                        
                        # Анализ аудиоконтента
                        audio_analysis = self.analyze_audio_content(full_path)
                        
                        # Объединение результатов - исправлена опечатка filename_bmp -> filename_bpm
                        final_bpm = filename_bmp if filename_bmp != 120 else audio_analysis["tempo"]
                        final_key = filename_key or audio_analysis["key"]
                        all_tags = list(set(filename_tags + audio_analysis["content_tags"]))
                        
                        # Определение категории
                        category = "loop" if duration > 8 else "oneshot"
                        if "loop" in filename_tags or "loop" in file.lower():
                            category = "loop"
                        
                        # Жанровая классификация по пути
                        path_lower = full_path.lower()
                        detected_genres = []
                        for genre, keywords in GENRE_KEYWORDS.items():
                            for keyword in keywords[:3]:  # Топ-3 ключевых слова
                                if keyword in path_lower:
                                    detected_genres.append(genre)
                                    break
                        
                        # Определение инструментальной роли
                        instrument_role = self._detect_instrument_role(file, audio_analysis)
                        
                        # Создаем SampleMetadata объект
                        metadata = SampleMetadata(
                            path=full_path,
                            filename=file,
                            duration=round(duration, 3),
                            tempo=round(final_bpm),
                            key=final_key,
                            tags=all_tags,
                            genres=detected_genres,
                            instrument_role=instrument_role,
                            spectral_centroid=audio_analysis["spectral_centroid"],
                            spectral_rolloff=audio_analysis["spectral_rolloff"],
                            zero_crossing_rate=audio_analysis["zero_crossing_rate"],
                            mfcc_features=audio_analysis["mfcc"],
                            chroma_features=audio_analysis["chroma"],
                            relative_path=os.path.relpath(full_path, self.sample_dir),
                            category=category,
                            quality_score=self._calculate_quality_score(audio_analysis, all_tags),
                            energy_level=audio_analysis["energy"],
                            brightness=audio_analysis["brightness"],
                            rhythmic_complexity=audio_analysis["rhythmic_complexity"]
                        )
                        
                        enhanced_index.append(metadata)
                        processed += 1
                        
                        if processed % 50 == 0:
                            logging.info(f"✅ Обработано: {processed} файлов")
                    
                    except Exception as e:
                        logging.warning(f"⚠️ Ошибка обработки {file}: {e}")
        
        logging.info(f"🎯 Расширенная индексация завершена: {len(enhanced_index)} сэмплов")
        return enhanced_index

    def _detect_instrument_role(self, filename: str, audio_analysis: Dict) -> Optional[str]:
        """Продвинутое определение инструментальной роли"""
        name_lower = filename.lower()
        
        # Поиск по ключевым словам в имени
        if any(word in name_lower for word in ['kick', 'bd', 'bassdrum']):
            return 'kick'
        elif any(word in name_lower for word in ['snare', 'sd']):
            return 'snare'
        elif any(word in name_lower for word in ['hihat', 'hh', 'hat']):
            return 'hihat'
        elif any(word in name_lower for word in ['bass', 'sub']):
            return 'bass'
        elif any(word in name_lower for word in ['lead', 'melody']):
            return 'lead'
        elif any(word in name_lower for word in ['pad', 'chord']):
            return 'pad'
        elif any(word in name_lower for word in ['fx', 'effect', 'riser']):
            return 'fx'
        
        # Классификация по аудиохарактеристикам
        energy = audio_analysis.get("energy", 0.5)
        brightness = audio_analysis.get("brightness", 0.0)
        centroid = audio_analysis.get("spectral_centroid", 0)
        zcr = audio_analysis.get("zero_crossing_rate", 0)
        
        # Более детальная классификация
        if centroid < 500:
            return "kick" if energy > 0.6 else "bass"
        elif centroid < 1500:
            return "snare" if energy > 0.5 and zcr > 0.1 else "bass"
        elif centroid < 4000:
            return "lead" if energy > 0.4 else "pad"
        else:
            return "hihat" if zcr > 0.2 else "fx"

    def _calculate_quality_score(self, audio_analysis: Dict, tags: List[str]) -> float:
        """Расчёт скора качества сэмпла"""
        score = 0.0
        
        # Бонус за наличие тегов
        if len(tags) > 0:
            score += 0.2
        if len(tags) > 2:
            score += 0.2
        
        # Качество аудио
        if audio_analysis.get("spectral_centroid", 0) > 0:
            score += 0.2
        
        # Энергетический баланс
        energy = audio_analysis.get("energy", 0)
        if 0.1 < energy < 0.9:  # Не слишком тихо и не клиппинг
            score += 0.2
        
        # Спектральная информация
        if audio_analysis.get("spectral_rolloff", 0) > 0:
            score += 0.2
        
        return min(1.0, score)

    def load_or_build_index(self):
        """Загрузка или создание индекса"""
        index_file = getattr(config, 'ENHANCED_INDEX_FILE', 'enhanced_sample_index.json')
        index_path = Path(self.sample_dir) / index_file
        semantic_cache_file = getattr(config, 'SEMANTIC_CACHE_FILE', 'semantic_cache.pkl')
        semantic_cache_path = Path(self.sample_dir) / semantic_cache_file
        
        if index_path.exists():
            self.logger.info(f"📂 Загружаем индекс сэмплов из {index_path}")
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.samples_index = []
                for item in data:
                    # Безопасное создание SampleMetadata
                    metadata = SampleMetadata(
                        path=item.get("path", ""),
                        filename=item.get("filename", ""),
                        duration=item.get("duration", 0.0),
                        tempo=item.get("tempo", 120),
                        key=item.get("key"),
                        tags=item.get("tags", []),
                        genres=item.get("genres", []),
                        instrument_role=item.get("instrument_role"),
                        spectral_centroid=item.get("spectral_centroid", 0.0),
                        spectral_rolloff=item.get("spectral_rolloff", 0.0),
                        zero_crossing_rate=item.get("zero_crossing_rate", 0.0),
                        mfcc_features=np.array(item.get("mfcc_features", [])),
                        chroma_features=np.array(item.get("chroma_features", [])),
                        relative_path=item.get("relative_path"),
                        category=item.get("category"),
                        quality_score=item.get("quality_score", 0.0),
                        energy_level=item.get("energy_level", 0.0),
                        brightness=item.get("brightness", 0.0),
                        rhythmic_complexity=item.get("rhythmic_complexity", 0.0)
                    )
                    self.samples_index.append(metadata)
                
                # Загружаем семантический кэш если доступен
                if SEMANTIC_AVAILABLE and semantic_cache_path.exists():
                    try:
                        with open(semantic_cache_path, 'rb') as f:
                            semantic_cache = pickle.load(f)
                        
                        # Применяем семантические embedding к сэмплам
                        embeddings = semantic_cache.get("embeddings", [])
                        clusters = semantic_cache.get("clusters", [])
                        
                        for i, sample in enumerate(self.samples_index):
                            if i < len(embeddings):
                                sample.semantic_embedding = embeddings[i]
                            if i < len(clusters):
                                sample.semantic_cluster = clusters[i]
                        
                        self.logger.info(f"📚 Загружены семантические данные для {len(embeddings)} сэмплов")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Ошибка загрузки семантического кэша: {e}")
                
                self.logger.info(f"✅ Загружено {len(self.samples_index)} сэмплов")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка загрузки индекса: {e}")
                self.build_semantic_index()
        else:
            self.logger.info("⚙️ Индекс не найден, строим новый")
            self.build_semantic_index()

    def build_semantic_index(self):
        """Построение полного семантического индекса"""
        self.logger.info("🧠 Построение семантического индекса сэмплов...")
        
        # Сначала строим базовый индекс
        self.samples_index = self.build_enhanced_index()
        
        # Затем добавляем семантические embedding если доступно
        if SEMANTIC_AVAILABLE and self.semantic_model and self.samples_index:
            self.logger.info("🧠 Генерация семантических embedding...")
            
            try:
                # Создаём семантические тексты
                semantic_texts = []
                for sample in self.samples_index:
                    semantic_text = self._create_semantic_text(sample)
                    semantic_texts.append(semantic_text)
                
                # Генерируем embeddings
                embeddings = self.semantic_model.encode(semantic_texts, show_progress_bar=True)
                
                # Кластеризация
                n_clusters = min(50, len(self.samples_index) // 10)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(embeddings)
                else:
                    clusters = [0] * len(embeddings)
                
                # Применяем к сэмплам
                for i, (sample, embedding, cluster) in enumerate(zip(self.samples_index, embeddings, clusters)):
                    sample.semantic_embedding = embedding
                    sample.semantic_cluster = int(cluster)
                
                # Сохраняем семантический кэш
                semantic_cache = {
                    "embeddings": embeddings,
                    "clusters": clusters,
                    "model_name": getattr(config, 'SAMPLE_MATCHING', {}).get("embedding_model", "default")
                }
                
                semantic_cache_file = getattr(config, 'SEMANTIC_CACHE_FILE', 'semantic_cache.pkl')
                semantic_cache_path = Path(self.sample_dir) / semantic_cache_file
                with open(semantic_cache_path, 'wb') as f:
                    pickle.dump(semantic_cache, f)
                
                self.logger.info(f"✅ Создано {len(embeddings)} семантических embedding, {n_clusters} кластеров")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка создания семантических embedding: {e}")
        
        # Сохраняем индекс
        self.save_index()
        
        self.logger.info(f"🎯 Семантический индекс построен: {len(self.samples_index)} сэмплов")

    def _create_semantic_text(self, sample: SampleMetadata) -> str:
        """Создание семантического текста для embedding"""
        text_parts = []
        
        # Добавляем теги
        text_parts.extend(sample.tags)
        
        # Добавляем жанры
        text_parts.extend(sample.genres)
        
        # Добавляем инструмент
        if sample.instrument_role:
            text_parts.append(sample.instrument_role)
        
        # Добавляем характеристики энергии
        if sample.energy_level > 0.7:
            text_parts.append("high-energy aggressive powerful")
        elif sample.energy_level < 0.3:
            text_parts.append("low-energy calm peaceful")
        
        # Добавляем спектральные характеристики
        if sample.brightness > 0.3:
            text_parts.append("bright")
        elif sample.brightness < 0.1:
            text_parts.append("dark")
        
        return " ".join(text_parts)

    def save_index(self):
        """Сохранение индекса в JSON"""
        index_file = getattr(config, 'ENHANCED_INDEX_FILE', 'enhanced_sample_index.json')
        index_path = Path(self.sample_dir) / index_file
        
        try:
            index_data = []
            for sample in self.samples_index:
                item = {
                    "path": sample.path,
                    "filename": sample.filename,
                    "duration": sample.duration,
                    "tempo": sample.tempo,
                    "key": sample.key,
                    "tags": sample.tags,
                    "genres": sample.genres,
                    "instrument_role": sample.instrument_role,
                    "spectral_centroid": sample.spectral_centroid,
                    "spectral_rolloff": sample.spectral_rolloff,
                    "zero_crossing_rate": sample.zero_crossing_rate,
                    "mfcc_features": sample.mfcc_features.tolist() if sample.mfcc_features.size > 0 else [],
                    "chroma_features": sample.chroma_features.tolist() if sample.chroma_features.size > 0 else [],
                    "relative_path": sample.relative_path,
                    "category": sample.category,
                    "quality_score": sample.quality_score,
                    "energy_level": sample.energy_level,
                    "brightness": sample.brightness,
                    "rhythmic_complexity": sample.rhythmic_complexity
                }
                index_data.append(item)
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Индекс сохранён: {index_path} ({len(index_data)} сэмплов)")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения индекса: {e}")

    async def find_samples(
        self,
        tags: List[str],
        instruments: Optional[List[str]] = None,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        energy: float = 0.5,
        max_results: int = 10,
        min_quality: float = 0.3
    ) -> List[Dict]:
        """
        Главная функция семантического поиска сэмплов
        
        Args:
            tags: Список тегов для поиска
            instruments: Приоритетные инструменты
            genre: Целевой жанр
            bpm: Целевой BPM
            energy: Уровень энергии (0-1)
            max_results: Максимум результатов
            min_quality: Минимальное качество (0-1)
        """
        start_time = time.time()
        self.performance_stats["queries"] += 1
        
        self.logger.info(f"🔍 Семантический поиск: теги={tags}, жанр={genre}, BPM={bpm}")
        
        # Создаём поисковый запрос
        search_query = {
            "tags": tags,
            "instruments": instruments or [],
            "genre": genre,
            "bpm": bpm,
            "energy": energy,
            "min_quality": min_quality
        }
        
        # Проверяем кэш
        query_hash = self._hash_query(search_query)
        if query_hash in self.embeddings_cache:
            self.performance_stats["cache_hits"] += 1
            self.logger.info("  💨 Результат из кэша")
            return self.embeddings_cache[query_hash][:max_results]
        
        # Выполняем поиск
        candidates = await self._search_candidates(search_query)
        scored_samples = await self._score_samples(candidates, search_query)
        
        # Фильтруем и сортируем
        filtered_samples = [
            sample for sample in scored_samples 
            if sample["score"] >= min_quality and sample["metadata"].quality_score >= min_quality
        ]
        
        # Сортируем по скору
        filtered_samples.sort(key=lambda x: x["score"], reverse=True)
        
        # Применяем диверсификацию результатов
        diversified_samples = self._diversify_results(filtered_samples, max_results)
        
        # Кэшируем результат
        self.embeddings_cache[query_hash] = diversified_samples
        
        # Обновляем статистику
        query_time = time.time() - start_time
        self.performance_stats["avg_query_time"] = (
            (self.performance_stats["avg_query_time"] * (self.performance_stats["queries"] - 1) + query_time) 
            / self.performance_stats["queries"]
        )
        
        self.logger.info(f"  ✅ Найдено {len(diversified_samples)} сэмплов за {query_time:.2f}с")
        
        return diversified_samples

    async def _search_candidates(self, query: Dict) -> List[SampleMetadata]:
        """Поиск кандидатов с предварительной фильтрацией"""
        candidates = []
        
        # Фильтрация по жанру
        if query["genre"]:
            genre_samples = [s for s in self.samples_index if query["genre"] in s.genres]
            if len(genre_samples) > 100:  # Если много сэмплов жанра, используем их
                base_samples = genre_samples
            else:
                base_samples = self.samples_index
        else:
            base_samples = self.samples_index
        
        # Фильтрация по BPM
        if query["bpm"]:
            tempo_tolerance = getattr(config, 'SAMPLE_MATCHING', {}).get("tempo_tolerance", 15)
            bpm_range = (query["bpm"] - tempo_tolerance, query["bpm"] + tempo_tolerance)
            base_samples = [
                s for s in base_samples 
                if bpm_range[0] <= s.tempo <= bpm_range[1]
            ]
        
        # Фильтрация по качеству
        min_quality = query["min_quality"]
        base_samples = [s for s in base_samples if s.quality_score >= min_quality]
        
        # Фильтрация по энергии (±0.3 от целевой)
        target_energy = query["energy"]
        energy_tolerance = 0.3
        base_samples = [
            s for s in base_samples 
            if abs(s.energy_level - target_energy) <= energy_tolerance
        ]
        
        self.logger.info(f"  📋 Кандидатов после фильтрации: {len(base_samples)}")
        
        return base_samples

    async def _score_samples(self, candidates: List[SampleMetadata], query: Dict) -> List[Dict]:
        """Скоринг сэмплов по семантическому и аудио-подобию"""
        scored_samples = []
        
        # Создаём семантический запрос
        semantic_query = self._create_semantic_query(query)
        
        for sample in candidates:
            score = await self._calculate_sample_score(sample, query, semantic_query)
            
            scored_samples.append({
                "metadata": sample,
                "score": score,
                "path": sample.path,
                "filename": sample.filename,
                "instrument_role": sample.instrument_role,
                "tags": sample.tags,
                "tempo": sample.tempo,
                "spectral_centroid": sample.spectral_centroid,
                "spectral_rolloff": sample.spectral_rolloff,
                "zero_crossing_rate": sample.zero_crossing_rate,
                "mfcc_features": sample.mfcc_features.tolist() if sample.mfcc_features.size > 0 else [],
                "chroma_features": sample.chroma_features.tolist() if sample.chroma_features.size > 0 else [],
                "quality_score": sample.quality_score,
                "energy_level": sample.energy_level,
                "brightness": sample.brightness,
                "rhythmic_complexity": sample.rhythmic_complexity,
                "semantic_cluster": sample.semantic_cluster
            })
        
        return scored_samples

    async def _calculate_sample_score(
        self, sample: SampleMetadata, query: Dict, semantic_query: str
    ) -> float:
        """Вычисление комплексного скора сэмпла"""
        score_components = {}
        
        # 1. Семантический скор (40%)
        if self.semantic_model and sample.semantic_embedding is not None:
            semantic_score = self._calculate_semantic_similarity(sample, semantic_query)
            score_components["semantic"] = semantic_score * 0.4
        else:
            # Fallback: скор по тегам
            tag_score = self._calculate_tag_similarity(sample.tags, query["tags"])
            score_components["semantic"] = tag_score * 0.4
        
        # 2. Темповое соответствие (15%)
        if query["bpm"]:
            tempo_score = self._calculate_tempo_similarity(sample.tempo, query["bpm"])
            score_components["tempo"] = tempo_score * 0.15
        else:
            score_components["tempo"] = 0.1  # Нейтральный скор если BPM не важен
        
        # 3. Жанровое соответствие (15%)
        if query["genre"]:
            genre_score = 1.0 if query["genre"] in sample.genres else 0.3
            score_components["genre"] = genre_score * 0.15
        else:
            score_components["genre"] = 0.1
        
        # 4. Энергетическое соответствие (10%)
        energy_score = 1.0 - abs(sample.energy_level - query["energy"])
        score_components["energy"] = max(0, energy_score) * 0.1
        
        # 5. Инструментальное соответствие (10%)
        instrument_score = 0.0
        if query["instruments"]:
            if sample.instrument_role in query["instruments"]:
                instrument_score = 1.0
            else:
                # Проверяем семантическое соответствие инструментов
                for instrument in query["instruments"]:
                    if any(instrument.lower() in tag.lower() for tag in sample.tags):
                        instrument_score = 0.7
                        break
        score_components["instrument"] = instrument_score * 0.1
        
        # 6. Качественный бонус (10%)
        quality_bonus = sample.quality_score * 0.1
        score_components["quality"] = quality_bonus
        
        # Итоговый скор
        total_score = sum(score_components.values())
        
        # Логирование для отладки (только для топ сэмплов)
        if total_score > 0.6:
            self.logger.debug(f"    🎯 {sample.filename}: {total_score:.2f} "
                            f"(sem:{score_components['semantic']:.2f}, "
                            f"tempo:{score_components['tempo']:.2f}, "
                            f"genre:{score_components['genre']:.2f})")
        
        return total_score

    def _calculate_semantic_similarity(self, sample: SampleMetadata, semantic_query: str) -> float:
        """Расчёт семантического сходства через SBERT"""
        if not self.semantic_model or sample.semantic_embedding is None:
            return 0.0
        
        try:
            # Энкодируем запрос
            query_embedding = self.semantic_model.encode([semantic_query])
            
            # Вычисляем cosine similarity
            similarity = cosine_similarity(
                query_embedding, 
                sample.semantic_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Ошибка семантического сравнения: {e}")
            return 0.0

    def _calculate_tag_similarity(self, sample_tags: List[str], query_tags: List[str]) -> float:
        """Fallback расчёт сходства по тегам"""
        if not query_tags or not sample_tags:
            return 0.0
        
        # Приводим к нижнему регистру для сравнения
        sample_tags_lower = [tag.lower() for tag in sample_tags]
        query_tags_lower = [tag.lower() for tag in query_tags]
        
        # Прямые совпадения
        direct_matches = len(set(sample_tags_lower) & set(query_tags_lower))
        
        # Частичные совпадения (подстроки)
        partial_matches = 0
        for sample_tag in sample_tags_lower:
            for query_tag in query_tags_lower:
                if query_tag in sample_tag or sample_tag in query_tag:
                    partial_matches += 0.5
        
        # Нормализация
        total_score = (direct_matches + partial_matches) / len(query_tags_lower)
        
        return min(1.0, total_score)

    def _calculate_tempo_similarity(self, sample_tempo: int, target_tempo: int) -> float:
        """Расчёт темпового соответствия"""
        tempo_diff = abs(sample_tempo - target_tempo)
        tolerance = getattr(config, 'SAMPLE_MATCHING', {}).get("tempo_tolerance", 15)
        
        if tempo_diff <= 3:
            return 1.0
        elif tempo_diff <= tolerance:
            return 1.0 - (tempo_diff / tolerance)
        else:
            return 0.0

    def _create_semantic_query(self, query: Dict) -> str:
        """Создание семантического запроса для SBERT"""
        query_parts = []
        
        # Добавляем теги
        if query["tags"]:
            query_parts.append(" ".join(query["tags"]))
        
        # Добавляем инструменты
        if query["instruments"]:
            query_parts.append(" ".join(query["instruments"]))
        
        # Добавляем жанр
        if query["genre"]:
            query_parts.append(query["genre"])
        
        # Добавляем характеристики энергии
        energy_level = query.get("energy", 0.5)
        if energy_level > 0.7:
            query_parts.append("energetic high-energy aggressive")
        elif energy_level < 0.3:
            query_parts.append("calm peaceful soft quiet")
        else:
            query_parts.append("medium-energy balanced")
        
        return " ".join(query_parts)

    def _diversify_results(self, samples: List[Dict], max_results: int) -> List[Dict]:
        """Диверсификация результатов для избежания повторов"""
        if len(samples) <= max_results:
            return samples
        
        diversified = []
        used_instruments = set()
        used_clusters = set()
        
        # Первый проход: берём лучшие сэмплы разных инструментов
        for sample in samples:
            if len(diversified) >= max_results:
                break
            
            instrument = sample.get("instrument_role")
            cluster = getattr(sample["metadata"], "semantic_cluster", None)
            
            # Предпочитаем разные инструменты и кластеры
            if instrument not in used_instruments or cluster not in used_clusters:
                diversified.append(sample)
                used_instruments.add(instrument)
                used_clusters.add(cluster)
        
        # Второй проход: добавляем оставшиеся лучшие сэмплы
        for sample in samples:
            if len(diversified) >= max_results:
                break
            if sample not in diversified:
                diversified.append(sample)
        
        return diversified[:max_results]

    def _hash_query(self, query: Dict) -> str:
        """Создание хэша для кэширования запросов"""
        import hashlib
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

    def get_statistics(self) -> Dict:
        """Получение статистики по индексу"""
        if not self.samples_index:
            return {"total_samples": 0}
        
        stats = {
            "total_samples": len(self.samples_index),
            "avg_quality": sum(s.quality_score for s in self.samples_index) / len(self.samples_index),
            "genre_distribution": {},
            "instrument_distribution": {},
            "tempo_ranges": {"slow": 0, "medium": 0, "fast": 0},
            "energy_distribution": {"low": 0, "medium": 0, "high": 0},
            "performance_stats": self.performance_stats
        }
        
        # Статистика по жанрам
        for sample in self.samples_index:
            for genre in sample.genres:
                stats["genre_distribution"][genre] = stats["genre_distribution"].get(genre, 0) + 1
        
        # Статистика по инструментам
        for sample in self.samples_index:
            if sample.instrument_role:
                role = sample.instrument_role
                stats["instrument_distribution"][role] = stats["instrument_distribution"].get(role, 0) + 1
        
        # Статистика по темпу
        for sample in self.samples_index:
            if sample.tempo < 100:
                stats["tempo_ranges"]["slow"] += 1
            elif sample.tempo < 140:
                stats["tempo_ranges"]["medium"] += 1
            else:
                stats["tempo_ranges"]["fast"] += 1
        
        # Статистика по энергии
        for sample in self.samples_index:
            if sample.energy_level < 0.4:
                stats["energy_distribution"]["low"] += 1
            elif sample.energy_level < 0.7:
                stats["energy_distribution"]["medium"] += 1
            else:
                stats["energy_distribution"]["high"] += 1
        
        return stats


# ===== СИСТЕМА ЭФФЕКТОВ =====

class EffectsProcessor:
    """Базовый класс для обработки эффектов"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """Применение эффекта к аудио"""
        raise NotImplementedError


class EQProcessor(EffectsProcessor):
    """Эквалайзер - трёхполосный EQ"""
    
    def __init__(self):
        super().__init__("EQ")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Применение EQ
        params: {"low": gain_db, "mid": gain_db, "high": gain_db}
        """
        try:
            processed = audio
            
            low_gain = params.get("low", 0)
            mid_gain = params.get("mid", 0) 
            high_gain = params.get("high", 0)
            
            # Низкие частоты (до 300Hz)
            if low_gain != 0:
                low_filtered = processed.low_pass_filter(300)
                if low_gain > 0:
                    low_filtered = low_filtered + low_gain
                else:
                    low_filtered = low_filtered.apply_gain(low_gain)
                
                high_pass = processed.high_pass_filter(300)
                processed = low_filtered.overlay(high_pass)
            
            # Высокие частоты (от 3000Hz)  
            if high_gain != 0:
                high_filtered = processed.high_pass_filter(3000)
                if high_gain > 0:
                    high_filtered = high_filtered + high_gain
                else:
                    high_filtered = high_filtered.apply_gain(high_gain)
                
                low_pass = processed.low_pass_filter(3000)
                processed = low_pass.overlay(high_filtered)
            
            # Средние частоты (симуляция)
            if mid_gain != 0:
                processed = processed + (mid_gain * 0.5)  # Простая симуляция
            
            self.logger.debug(f"EQ applied: L{low_gain:+.1f} M{mid_gain:+.1f} H{high_gain:+.1f}")
            return processed
            
        except Exception as e:
            self.logger.error(f"EQ processing error: {e}")
            return audio


class CompressorProcessor(EffectsProcessor):
    """Компрессор динамического диапазона"""
    
    def __init__(self):
        super().__init__("Compressor")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Применение компрессии
        params: {"ratio": float, "threshold": db, "attack": ms, "release": ms}
        """
        try:
            ratio = params.get("ratio", 2.0)
            threshold = params.get("threshold", -12)
            
            # Используем встроенную компрессию pydub как базу
            if ratio > 1:
                # Мягкая компрессия через нормализацию и лимитирование
                peak_level = audio.max_dBFS
                
                if peak_level > threshold:
                    over_threshold = peak_level - threshold
                    reduction = over_threshold - (over_threshold / ratio)
                    processed = audio - reduction
                    
                    self.logger.debug(f"Compression: {ratio}:1, reduction {reduction:.1f}dB")
                    return processed
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Compression processing error: {e}")
            return audio


class ReverbProcessor(EffectsProcessor):
    """Реверберация (симуляция через задержку и фильтрацию)"""
    
    def __init__(self):
        super().__init__("Reverb")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Применение реверба
        params: {"room_size": 0-1, "wet_level": 0-1, "type": "hall/room/plate"}
        """
        try:
            room_size = params.get("room_size", 0.3)
            wet_level = params.get("wet_level", 0.2)
            reverb_type = params.get("type", "room")
            
            if wet_level <= 0:
                return audio
            
            # Создаём простую симуляцию реверба через задержки
            delays = []
            
            if reverb_type == "hall":
                delays = [(50, 0.3), (120, 0.2), (250, 0.15), (400, 0.1)]
            elif reverb_type == "plate":
                delays = [(20, 0.4), (60, 0.3), (140, 0.2)]
            else:  # room
                delays = [(30, 0.25), (80, 0.15), (150, 0.1)]
            
            # Масштабируем задержки по размеру комнаты
            scaled_delays = [(int(delay * (0.5 + room_size)), gain * wet_level) 
                           for delay, gain in delays]
            
            reverb_audio = AudioSegment.silent(duration=0)
            
            for delay_ms, gain in scaled_delays:
                if delay_ms > 0 and gain > 0:
                    delayed = AudioSegment.silent(duration=delay_ms) + audio
                    delayed = delayed.apply_gain(-20 * (1 - gain))  # Затухание
                    
                    if len(reverb_audio) == 0:
                        reverb_audio = delayed
                    else:
                        reverb_audio = reverb_audio.overlay(delayed)
            
            # Смешиваем оригинал с ревербом
            if len(reverb_audio) > 0:
                # Приводим к одной длине
                max_len = max(len(audio), len(reverb_audio))
                if len(audio) < max_len:
                    audio = audio + AudioSegment.silent(duration=max_len - len(audio))
                if len(reverb_audio) < max_len:
                    reverb_audio = reverb_audio + AudioSegment.silent(duration=max_len - len(reverb_audio))
                
                # Фильтруем реверб для более естественного звука
                reverb_audio = reverb_audio.low_pass_filter(8000)
                
                result = audio.overlay(reverb_audio)
                
                self.logger.debug(f"Reverb applied: {reverb_type}, size {room_size}, wet {wet_level}")
                return result
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Reverb processing error: {e}")
            return audio


class SaturationProcessor(EffectsProcessor):
    """Гармоническое насыщение (симуляция аналогового тепла)"""
    
    def __init__(self):
        super().__init__("Saturation")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Применение насыщения
        params: {"amount": 0-1, "type": "tube/tape/transistor", "warmth": 0-1}
        """
        try:
            amount = params.get("amount", 0.0)
            saturation_type = params.get("type", "tube")
            warmth = params.get("warmth", 0.0)
            
            if amount <= 0:
                return audio
            
            processed = audio
            
            # Симуляция насыщения через лёгкое поднятие низких частот и компрессию
            if saturation_type == "tube":
                # Ламповое насыщение - тёплое, мягкое
                if warmth > 0:
                    # Поднимаем низкие частоты
                    low_boost = warmth * 2  # dB
                    low_freq = processed.low_pass_filter(800) + low_boost
                    high_freq = processed.high_pass_filter(800)
                    processed = low_freq.overlay(high_freq)
                    
                # Мягкая компрессия
                if processed.max_dBFS > -6:
                    soft_compress = amount * 2
                    processed = processed - soft_compress
                    
            elif saturation_type == "tape":
                # Ленточное насыщение - винтажное
                processed = processed.high_pass_filter(30).low_pass_filter(15000)  # Винтажная полоса
                processed = processed + (amount * 1.5)  # Лёгкий подъём
                
            elif saturation_type == "transistor":
                # Транзисторное насыщение - более агрессивное
                if processed.max_dBFS > -3:
                    hard_compress = amount * 3
                    processed = processed - hard_compress
            
            self.logger.debug(f"Saturation applied: {saturation_type}, amount {amount}, warmth {warmth}")
            return processed
            
        except Exception as e:
            self.logger.error(f"Saturation processing error: {e}")
            return audio


class StereoProcessor(EffectsProcessor):
    """Стерео обработка и расширение"""
    
    def __init__(self):
        super().__init__("Stereo")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Стерео обработка
        params: {"width": 0.5-2.0, "imaging": "natural/enhanced/mono"}
        """
        try:
            width = params.get("width", 1.0)
            imaging = params.get("imaging", "natural")
            
            if audio.channels < 2:
                # Преобразуем в стерео если нужно
                audio = audio.set_channels(2)
            
            processed = audio
            
            if imaging == "mono":
                # Моносигнал
                processed = processed.set_channels(1).set_channels(2)
                
            elif imaging == "enhanced" and width != 1.0:
                if width > 1.0:
                    # Расширение стерео
                    # Простая симуляция - добавляем задержку между каналами
                    self.logger.debug(f"Stereo widening: {width:.1f}x")
                    
                elif width < 1.0:
                    # Сужение стерео - микс с моно
                    mono_component = processed.set_channels(1).set_channels(2)
                    mono_gain = (1.0 - width) * -6  # dB 
                    processed = processed.overlay(mono_component + mono_gain)
                    self.logger.debug(f"Stereo narrowing: {width:.1f}x")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Stereo processing error: {e}")
            return audio


class LimiterProcessor(EffectsProcessor):
    """Лимитер - предотвращение клиппинга"""
    
    def __init__(self):
        super().__init__("Limiter")
    
    async def process(self, audio: AudioSegment, params: Dict) -> AudioSegment:
        """
        Применение лимитера
        params: {"threshold": db, "ceiling": db, "release": ms}
        """
        try:
            threshold = params.get("threshold", -3)
            ceiling = params.get("ceiling", -0.1)
            
            peak_level = audio.max_dBFS
            
            if peak_level > threshold:
                # Мягкое лимитирование
                over_threshold = peak_level - threshold
                if over_threshold > 0:
                    # Рассчитываем необходимое снижение
                    target_peak = min(ceiling, threshold)
                    required_reduction = peak_level - target_peak
                    
                    # Применяем лимитирование
                    processed = audio - required_reduction
                    
                    self.logger.debug(f"Limiting: threshold {threshold}dB, ceiling {ceiling}dB, "
                                    f"reduction {required_reduction:.1f}dB")
                    return processed
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Limiter processing error: {e}")
            return audio


class EffectsChain:
    """Цепочка эффектов с возможностью загрузки конфигурации из JSON"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Регистрируем доступные процессоры
        self.processors = {
            "eq": EQProcessor(),
            "compressor": CompressorProcessor(),
            "reverb": ReverbProcessor(),
            "saturation": SaturationProcessor(),
            "stereo": StereoProcessor(),
            "limiter": LimiterProcessor()
        }
        
        self.logger.info(f"✅ Effects chain initialized: {list(self.processors.keys())}")
    
    async def apply_effects(
        self, 
        audio: AudioSegment, 
        effects_config: Dict,
        genre_info: Optional[Dict] = None
    ) -> AudioSegment:
        """
        Применение цепочки эффектов
        
        effects_config format:
        {
            "eq": {"low": 2, "mid": 0, "high": 1},
            "compressor": {"ratio": 3.0, "threshold": -10},
            "reverb": {"room_size": 0.3, "wet_level": 0.15}
        }
        """
        if not effects_config:
            return audio
        
        processed = audio
        applied_effects = []
        
        # Применяем эффекты в определённом порядке
        effect_order = ["eq", "compressor", "saturation", "reverb", "stereo", "limiter"]
        
        for effect_name in effect_order:
            if effect_name in effects_config:
                effect_params = effects_config[effect_name]
                
                if effect_name in self.processors:
                    try:
                        processed = await self.processors[effect_name].process(processed, effect_params)
                        applied_effects.append(effect_name)
                        
                    except Exception as e:
                        self.logger.error(f"Error applying {effect_name}: {e}")
        
        self.logger.info(f"✨ Applied effects: {', '.join(applied_effects)}")
        return processed
    
    async def mix_layers(
        self,
        base_layer: bytes,
        stem_layers: Dict[str, bytes], 
        mix_settings: Dict,
        genre_info: Dict
    ) -> AudioSegment:
        """
        Микширование базовой дорожки со стемами
        
        Args:
            base_layer: Основная дорожка от MusicGen
            stem_layers: Словарь стемов {instrument: audio_data}
            mix_settings: Настройки микса для жанра
            genre_info: Информация о жанре
        """
        try:
            # Конвертируем байты в AudioSegment
            # В реальной реализации здесь будет загрузка из bytes
            # Временная заглушка:
            base_audio = AudioSegment.silent(duration=60000)  # 60 секунд
            
            # Применяем уровень базовой дорожки
            base_level = mix_settings.get("base_level", -3)
            mixed = base_audio + base_level
            
            # Добавляем стемы
            stems_level = mix_settings.get("stems_level", -6)
            
            for instrument, stem_data in stem_layers.items():
                # Конвертируем stem_data в AudioSegment
                # В реальной реализации здесь будет загрузка из bytes
                stem_audio = AudioSegment.silent(duration=len(mixed))
                
                # Применяем уровень стема
                stem_audio = stem_audio + stems_level
                
                # Применяем инструмент-специфичные настройки
                instrument_settings = self._get_instrument_mix_settings(instrument, genre_info)
                if instrument_settings:
                    stem_audio = await self.apply_effects(stem_audio, instrument_settings)
                
                # Микшируем
                mixed = mixed.overlay(stem_audio)
                
                self.logger.debug(f"  🎛️ Mixed {instrument}: {stems_level:+.1f}dB")
            
            self.logger.info(f"🎚️ Mixed base + {len(stem_layers)} stems")
            return mixed
            
        except Exception as e:
            self.logger.error(f"❌ Mixing error: {e}")
            # Возвращаем базовую дорожку в случае ошибки
            return base_audio
    
    def _get_instrument_mix_settings(self, instrument: str, genre_info: Dict) -> Optional[Dict]:
        """Получение настроек микса для конкретного инструмента"""
        genre = genre_info.get("name", "")
        
        instrument_effects = {
            "trap": {
                "kick": {"eq": {"low": 3, "mid": 0, "high": -1}},
                "snare": {"eq": {"low": -1, "mid": 2, "high": 4}, "compressor": {"ratio": 2.5}},
                "hihat": {"eq": {"low": -3, "mid": 0, "high": 2}},
                "bass": {"eq": {"low": 2, "mid": -1, "high": -2}, "saturation": {"amount": 0.2}}
            },
            "lofi": {
                "kick": {"eq": {"low": 1, "mid": 0, "high": -3}, "saturation": {"amount": 0.4, "type": "tape"}},
                "snare": {"eq": {"low": 0, "mid": -1, "high": -2}},
                "piano": {"reverb": {"room_size": 0.2, "wet_level": 0.3}, "saturation": {"amount": 0.3, "warmth": 0.5}}
            },
            "house": {
                "kick": {"eq": {"low": 2, "mid": 0, "high": 0}, "compressor": {"ratio": 3.0, "threshold": -8}},
                "bass": {"eq": {"low": 1, "mid": 0, "high": -1}},
                "lead": {"reverb": {"room_size": 0.4, "wet_level": 0.2}}
            },
            "ambient": {
                "pad": {"reverb": {"room_size": 0.8, "wet_level": 0.5, "type": "hall"}},
                "lead": {"reverb": {"room_size": 0.6, "wet_level": 0.4}},
                "fx": {"stereo": {"width": 1.5, "imaging": "enhanced"}}
            }
        }
        
        return instrument_effects.get(genre, {}).get(instrument)
    
    def load_effects_preset(self, preset_path: str) -> Optional[Dict]:
        """Загрузка пресета эффектов из JSON файла"""
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
            
            self.logger.info(f"📋 Loaded effects preset: {preset_path}")
            return preset
            
        except Exception as e:
            self.logger.error(f"❌ Error loading effects preset {preset_path}: {e}")
            return None
    
    def save_effects_preset(self, effects_config: Dict, preset_path: str) -> bool:
        """Сохранение пресета эффектов в JSON файл"""
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(effects_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved effects preset: {preset_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving effects preset {preset_path}: {e}")
            return False


# ===== УТИЛИТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

def create_genre_specific_effects(genre: str) -> Dict:
    """Создание жанро-специфичных настроек эффектов"""
    effects_templates = {
        "trap": {
            "eq": {"low": 2, "mid": 0, "high": 1},
            "compressor": {"ratio": 2.5, "threshold": -10},
            "saturation": {"amount": 0.15, "type": "transistor"}
        },
        "lofi": {
            "eq": {"low": 1, "mid": -1, "high": -3},
            "saturation": {"amount": 0.4, "type": "tape", "warmth": 0.6},
            "reverb": {"room_size": 0.3, "wet_level": 0.2, "type": "room"}
        },
        "house": {
            "eq": {"low": 1, "mid": 0, "high": 2},
            "compressor": {"ratio": 3.0, "threshold": -8},
            "reverb": {"room_size": 0.4, "wet_level": 0.15, "type": "hall"}
        },
        "ambient": {
            "eq": {"low": 0, "mid": 0, "high": 1},
            "reverb": {"room_size": 0.8, "wet_level": 0.6, "type": "hall"},
            "stereo": {"width": 1.4, "imaging": "enhanced"}
        },
        "dnb": {
            "eq": {"low": 3, "mid": 1, "high": 2},
            "compressor": {"ratio": 4.0, "threshold": -6},
            "saturation": {"amount": 0.2, "type": "transistor"}
        },
        "dubstep": {
            "eq": {"low": 4, "mid": 0, "high": 1},
            "compressor": {"ratio": 5.0, "threshold": -4},
            "saturation": {"amount": 0.3, "type": "transistor"},
            "limiter": {"threshold": -1, "ceiling": -0.1}
        }
    }
    
    return effects_templates.get(genre, {
        "eq": {"low": 0, "mid": 0, "high": 0},
        "compressor": {"ratio": 2.0, "threshold": -12}
    })


def validate_audio_file(filepath: str) -> bool:
    """Проверка валидности аудиофайла"""
    try:
        # Проверяем существование файла
        if not os.path.exists(filepath):
            return False
        
        # Проверяем размер файла
        file_size = os.path.getsize(filepath)
        if file_size < 1024:  # Меньше 1KB - подозрительно
            return False
        
        # Пробуем загрузить файл
        try:
            audio = AudioSegment.from_file(filepath)
            duration = len(audio) / 1000
            
            # Проверяем разумную длительность
            if duration < 0.1 or duration > 600:  # От 0.1 сек до 10 минут
                return False
                
            return True
            
        except:
            # Fallback через librosa
            y, sr = librosa.load(filepath, duration=1)  # Загружаем только первую секунду
            return len(y) > 0
            
    except Exception:
        return False


def extract_bpm_from_filename(filename: str) -> Optional[int]:
    """Извлечение BPM из имени файла с улучшенным распознаванием"""
    patterns = [
        r'(\d{2,3})\s*bpm',
        r'(\d{2,3})\s*beats',
        r'tempo[\s_-]*(\d{2,3})',
        r'(\d{2,3})[\s_-]*beat',
        r'(\d{2,3})[\s_-]*bpm',
        r'bpm[\s_-]*(\d{2,3})',
        r'(\d{2,3})[\s_-]*tempo'
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            bpm = int(match.group(1))
            # Валидация разумных значений BPM
            if 60 <= bpm <= 200:
                return bpm
    
    return None


def extract_key_from_filename(filename: str) -> Optional[str]:
    """Извлечение тональности из имени файла"""
    patterns = [
        r'\b([a-g][#b]?)\s*m(?:inor|in)?\b',
        r'\b([a-g][#b]?)\s*maj(?:or)?\b',
        r'\b([a-g][#b]?)[\s_-]*(?:key|scale)\b',
        r'key[\s_-]*([a-g][#b]?)\b'
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            key = match.group(1).upper()
            # Нормализуем обозначения
            key = key.replace('B', 'b')  # Бемоль
            return key
    
    return None


async def batch_process_samples(
    sample_paths: List[str], 
    processor_func, 
    max_workers: int = 4,
    batch_size: int = 50
) -> List[Any]:
    """Пакетная обработка сэмплов с контролем ресурсов"""
    results = []
    
    # Разбиваем на батчи
    for i in range(0, len(sample_paths), batch_size):
        batch = sample_paths[i:i + batch_size]
        
        # Обрабатываем батч параллельно
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: [processor_func(path) for path in batch]
            )
            
        results.extend(batch_results)
        
        # Небольшая пауза между батчами для предотвращения перегрузки
        await asyncio.sleep(0.1)
    
    return results


class SampleQualityAnalyzer:
    """Анализатор качества сэмплов"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityAnalyzer")
    
    def analyze_sample_quality(self, filepath: str, audio_analysis: Dict) -> Dict:
        """Комплексный анализ качества сэмпла"""
        quality_metrics = {
            "overall_score": 0.0,
            "audio_quality": 0.0,
            "metadata_completeness": 0.0,
            "spectral_quality": 0.0,
            "dynamic_range": 0.0,
            "issues": []
        }
        
        try:
            # Анализ аудио качества
            audio_quality = self._analyze_audio_quality(audio_analysis)
            quality_metrics["audio_quality"] = audio_quality["score"]
            quality_metrics["issues"].extend(audio_quality["issues"])
            
            # Анализ полноты метаданных
            metadata_quality = self._analyze_metadata_completeness(filepath, audio_analysis)
            quality_metrics["metadata_completeness"] = metadata_quality["score"]
            
            # Спектральный анализ
            spectral_quality = self._analyze_spectral_quality(audio_analysis)
            quality_metrics["spectral_quality"] = spectral_quality["score"]
            quality_metrics["issues"].extend(spectral_quality["issues"])
            
            # Динамический диапазон
            dynamic_range = self._analyze_dynamic_range(audio_analysis)
            quality_metrics["dynamic_range"] = dynamic_range["score"]
            quality_metrics["issues"].extend(dynamic_range["issues"])
            
            # Общий скор (взвешенное среднее)
            weights = [0.4, 0.2, 0.2, 0.2]  # аудио, метаданные, спектр, динамика
            scores = [
                quality_metrics["audio_quality"],
                quality_metrics["metadata_completeness"],
                quality_metrics["spectral_quality"],
                quality_metrics["dynamic_range"]
            ]
            
            quality_metrics["overall_score"] = sum(w * s for w, s in zip(weights, scores))
            
        except Exception as e:
            self.logger.error(f"Quality analysis error for {filepath}: {e}")
            quality_metrics["issues"].append(f"Analysis error: {e}")
        
        return quality_metrics
    
    def _analyze_audio_quality(self, audio_analysis: Dict) -> Dict:
        """Анализ качества аудиосигнала"""
        score = 1.0
        issues = []
        
        # Проверка уровня энергии
        energy = audio_analysis.get("energy", 0.5)
        if energy < 0.05:
            score -= 0.3
            issues.append("Very low energy level")
        elif energy > 0.95:
            score -= 0.2
            issues.append("Possible clipping detected")
        
        # Проверка спектрального содержимого
        spectral_centroid = audio_analysis.get("spectral_centroid", 0)
        if spectral_centroid == 0:
            score -= 0.4
            issues.append("No spectral content detected")
        elif spectral_centroid < 100:
            score -= 0.2
            issues.append("Very low spectral content")
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _analyze_metadata_completeness(self, filepath: str, audio_analysis: Dict) -> Dict:
        """Анализ полноты метаданных"""
        score = 0.0
        
        # Бонус за BPM информацию
        if extract_bpm_from_filename(os.path.basename(filepath)):
            score += 0.3
        if audio_analysis.get("tempo", 120) != 120:  # Не дефолтное значение
            score += 0.2
        
        # Бонус за тональность
        if extract_key_from_filename(os.path.basename(filepath)):
            score += 0.2
        if audio_analysis.get("key"):
            score += 0.1
        
        # Бонус за описательное имя файла
        filename = os.path.basename(filepath).lower()
        if len([word for word in filename.split() if len(word) > 2]) >= 2:
            score += 0.2
        
        return {"score": min(1.0, score)}
    
    def _analyze_spectral_quality(self, audio_analysis: Dict) -> Dict:
        """Анализ спектрального качества"""
        score = 1.0
        issues = []
        
        # Проверка спектральных характеристик
        spectral_rolloff = audio_analysis.get("spectral_rolloff", 0)
        spectral_centroid = audio_analysis.get("spectral_centroid", 0)
        
        if spectral_rolloff == 0 or spectral_centroid == 0:
            score -= 0.5
            issues.append("Missing spectral analysis data")
        else:
            # Проверка сбалансированности спектра
            if spectral_centroid > spectral_rolloff * 0.8:
                score -= 0.2
                issues.append("Unbalanced spectral distribution")
        
        # Проверка zero crossing rate
        zcr = audio_analysis.get("zero_crossing_rate", 0)
        if zcr > 0.3:
            score -= 0.1
            issues.append("High zero crossing rate (possible noise)")
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _analyze_dynamic_range(self, audio_analysis: Dict) -> Dict:
        """Анализ динамического диапазона"""
        score = 1.0
        issues = []
        
        # Анализ на основе энергии и спектральных характеристик
        energy = audio_analysis.get("energy", 0.5)
        rhythmic_complexity = audio_analysis.get("rhythmic_complexity", 0)
        
        # Очень высокая энергия может указывать на сжатый динамический диапазон
        if energy > 0.9:
            score -= 0.3
            issues.append("Possibly over-compressed (limited dynamic range)")
        
        # Очень низкая ритмическая сложность в энергичном треке
        if energy > 0.6 and rhythmic_complexity < 1.0:
            score -= 0.1
            issues.append("Low rhythmic complexity for energetic content")
        
        return {"score": max(0.0, score), "issues": issues}


# ===== ЭКСПОРТ ОСНОВНЫХ КЛАССОВ =====
__all__ = [
    'SemanticSampleEngine',
    'EffectsChain', 
    'SampleMetadata',
    'SampleQualityAnalyzer',
    'create_genre_specific_effects',
    'validate_audio_file',
    'extract_bpm_from_filename',
    'extract_key_from_filename',
    'batch_process_samples'
]