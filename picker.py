# WaveDream Enhanced - Полная переработка системы подбора сэмплов
import os
import json
import random
import re
import logging
import librosa
import numpy as np
from collections import defaultdict, Counter
from pydub import AudioSegment
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class EnhancedSamplePicker:
    """Улучшенная система подбора сэмплов с AI-анализом и семантическим поиском"""
    
    def __init__(self, sample_dir, index_file="enhanced_sample_index.json"):
        self.sample_dir = sample_dir
        self.index_file = os.path.join(sample_dir, index_file)

        
        # Расширенная база тегов с семантическими связями
        self.semantic_tags = {
            # Percussion семантика
            "kick": ["kick", "bd", "bass_drum", "thump", "punch", "boom", "sub_kick"],
            "snare": ["snare", "snr", "crack", "snap", "clap", "rim", "backbeat"],
            "hihat": ["hat", "hh", "hi_hat", "closed_hat", "open_hat", "cymbal"],
            "clap": ["clap", "snap", "handclap", "finger_snap"],
            "percussion": ["perc", "percussion", "bongo", "conga", "shaker", "tambourine"],
            
            # Bass семантика
            "bass": ["bass", "sub", "low", "808", "reese", "growl", "wobble"],
            "808": ["808", "sub_bass", "kick_bass", "trap_bass", "sliding_bass"],
            
            # Melodic семантика
            "lead": ["lead", "melody", "synth", "arps", "keys", "piano", "guitar"],
            "pad": ["pad", "strings", "ambient", "texture", "drone", "atmosphere"],
            "piano": ["piano", "keys", "electric_piano", "rhodes", "organ"],
            "guitar": ["guitar", "strum", "pluck", "acoustic", "electric"],
            
            # Vocal семантика
            "vocal": ["vocal", "voice", "vox", "choir", "chant", "rap", "sung"],
            "vocal_chop": ["vocal_chop", "chop", "stutter", "slice"],
            
            # FX семантика
            "fx": ["fx", "effect", "sweep", "riser", "impact", "whoosh", "transition"],
            "riser": ["riser", "sweep", "build", "tension", "uplifter"],
            
            # Genre-specific
            "trap": ["trap", "drill", "dark", "hard", "aggressive"],
            "house": ["house", "groove", "funky", "disco", "four_on_floor"],
            "techno": ["techno", "industrial", "minimal", "hypnotic"],
            "ambient": ["ambient", "chill", "relax", "meditation", "space"]
        }
        
        # Жанровые паттерны для BPM коррекции
        self.genre_bpm_ranges = {
            "trap": (130, 170),
            "dnb": (160, 180), 
            "house": (118, 128),
            "techno": (120, 135),
            "ambient": (60, 90),
            "lofi": (60, 85)
        }

        self.index = self.load_or_build_index()

    def analyze_filename_advanced(self, filename):
        """Продвинутый анализ имени файла с извлечением BPM, тональности и тегов"""
        name = filename.lower()
        tags = set()
        bpm = 120
        key = None
        
        # Извлечение BPM из имени файла
        bpm_patterns = [
            r'(\d{2,3})bpm',
            r'(\d{2,3})_bpm', 
            r'bpm(\d{2,3})',
            r'(\d{2,3})beats',
            r'tempo_?(\d{2,3})'
        ]
        
        for pattern in bpm_patterns:
            match = re.search(pattern, name)
            if match:
                bpm = int(match.group(1))
                break
        
        # Извлечение тональности
        key_patterns = [
            r'\b([a-g]#?)\s*m(?:inor)?\b',
            r'\b([a-g]#?)\s*maj(?:or)?\b',
            r'key_([a-g]#?)',
            r'in_([a-g]#?)'
        ]
        
        for pattern in key_patterns:
            match = re.search(pattern, name)
            if match:
                key = match.group(1).upper()
                break
        
        # Семантический анализ названия
        for tag_category, keywords in self.semantic_tags.items():
            for keyword in keywords:
                if keyword in name:
                    tags.add(tag_category)
                    
        # Дополнительные паттерны для жанров
        genre_patterns = {
            "trap": ["trap", "drill", "phonk", "memphis", "dark", "hard", "aggressive"],
            "house": ["house", "groove", "funky", "disco", "dance", "club"],
            "techno": ["techno", "tech", "minimal", "industrial", "berlin"],
            "dnb": ["dnb", "drum", "bass", "jungle", "break", "neuro"],
            "ambient": ["ambient", "chill", "space", "drone", "meditation"],
            "lofi": ["lofi", "lo-fi", "jazzy", "vinyl", "dusty", "vintage"]
        }
        
        for genre, patterns in genre_patterns.items():
            if any(p in name for p in patterns):
                tags.add(genre)
        
        # Инструментальные паттерны
        instrument_patterns = {
            "kick": ["kick", "bd", "bass_drum", "thump"],
            "snare": ["snare", "snr", "crack", "rim"],
            "hihat": ["hat", "hh", "hi_hat", "cymbal"],
            "bass": ["bass", "sub", "low", "808"],
            "lead": ["lead", "melody", "synth", "keys"],
            "pad": ["pad", "string", "ambient", "texture"],
            "vocal": ["vocal", "voice", "vox", "choir"],
            "fx": ["fx", "sweep", "riser", "impact", "whoosh"]
        }
        
        for instrument, patterns in instrument_patterns.items():
            if any(p in name for p in patterns):
                tags.add(instrument)
        
        return list(tags), bpm, key

    def analyze_audio_content(self, file_path, max_duration=10):
        """Анализ аудиоконтента для автоматического извлечения характеристик"""
        try:
            y, sr = librosa.load(file_path, duration=max_duration, sr=22050)
            
            # Анализ темпа
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
            
            # Анализ тональности  
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            if chroma.size > 0:
                chroma_mean = chroma.mean(axis=1)
                key_idx = chroma_mean.argmax()
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = keys[key_idx]
            else:
                key = None
            
            # Анализ спектральных характеристик
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
            
            # Определение типа инструмента по спектральным характеристикам
            content_tags = []
            
            # Низкочастотный контент (бас, кик)
            if spectral_centroid < 1000:
                content_tags.extend(["bass", "kick", "sub"])
            
            # Высокочастотный контент (хэты, тарелки)
            elif spectral_centroid > 8000:
                content_tags.extend(["hihat", "cymbal", "fx"])
                
            # Средние частоты (снейр, мелодия)
            elif 2000 < spectral_centroid < 6000:
                content_tags.extend(["snare", "lead", "melody"])
            
            # Анализ ритмичности
            if zero_crossing_rate > 0.1:
                content_tags.extend(["percussion", "rhythmic"])
            
            # Определение лупов vs one-shots
            if len(y) / sr > 8:  # Длиннее 8 секунд = вероятно луп
                content_tags.append("loop")
            else:
                content_tags.append("oneshot")
                
            return {
                "tempo": max(60, min(200, tempo)),  # Ограничиваем разумными пределами
                "key": key,
                "content_tags": content_tags,
                "spectral_centroid": float(spectral_centroid),
                "brightness": float(spectral_rolloff / sr)
            }
            
        except Exception as e:
            logging.warning(f"Ошибка анализа {file_path}: {e}")
            return {"tempo": 120, "key": None, "content_tags": [], "spectral_centroid": 0, "brightness": 0}

    def build_enhanced_index(self):
        """Построение расширенного индекса с AI-анализом"""
        logging.info("🔍 Начинаю расширенную индексацию сэмплов...")
        
        enhanced_index = []
        processed = 0
        
        for root, _, files in os.walk(self.sample_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.aiff', '.flac')):
                    full_path = os.path.join(root, file)
                    
                    try:
                        # Базовая информация
                        audio = AudioSegment.from_file(full_path)
                        duration = len(audio) / 1000
                        
                        # Анализ имени файла
                        filename_tags, filename_bpm, filename_key = self.analyze_filename_advanced(file)
                        
                        # Анализ аудиоконтента
                        audio_analysis = self.analyze_audio_content(full_path)
                        
                        # Объединение результатов
                        final_bpm = filename_bpm if filename_bpm != 120 else audio_analysis["tempo"]
                        final_key = filename_key or audio_analysis["key"]
                        
                        all_tags = list(set(filename_tags + audio_analysis["content_tags"]))
                        
                        # Определение категории
                        category = "loop" if duration > 8 else "oneshot"
                        if "loop" in filename_tags or "loop" in file.lower():
                            category = "loop"
                        
                        entry = {
                            "path": full_path,
                            "filename": file,
                            "tempo": round(final_bpm),
                            "duration": round(duration, 3),
                            "key": final_key,
                            "category": category,
                            "tags": all_tags,
                            "spectral_centroid": audio_analysis["spectral_centroid"],
                            "brightness": audio_analysis["brightness"],
                            "relative_path": os.path.relpath(full_path, self.sample_dir)
                        }
                        
                        enhanced_index.append(entry)
                        processed += 1
                        
                        if processed % 100 == 0:
                            logging.info(f"✅ Обработано: {processed} файлов")
                            
                    except Exception as e:
                        logging.warning(f"⚠️ Ошибка обработки {file}: {e}")
        
        logging.info(f"🎯 Индексация завершена: {len(enhanced_index)} сэмплов")
        return enhanced_index

    def load_or_build_index(self):
        """Загрузка или создание индекса"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                logging.info(f"📚 Загружен индекс: {len(index)} сэмплов")
                return index
            except Exception as e:
                logging.warning(f"⚠️ Ошибка загрузки индекса: {e}")
        
        # Создаём новый индекс
        index = self.build_enhanced_index()
        self.save_index(index)
        return index

    def save_index(self, index):
        """Сохранение индекса"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        logging.info(f"💾 Индекс сохранён: {self.index_file}")

    def find_semantic_matches(self, query_tag, sample_tags):
        """Семантический поиск совпадений тегов"""
        if not sample_tags:
            return 0
            
        # Прямое совпадение
        if query_tag in sample_tags:
            return 10
        
        # Семантические совпадения
        score = 0
        query_synonyms = self.semantic_tags.get(query_tag, [query_tag])
        
        for synonym in query_synonyms:
            for sample_tag in sample_tags:
                if synonym in sample_tag or sample_tag in synonym:
                    score += 5
                elif self.fuzzy_match(synonym, sample_tag):
                    score += 3
        
        return score

    def fuzzy_match(self, a, b, threshold=0.7):
        """Нечёткое сравнение строк"""
        a, b = a.lower(), b.lower()
        if a == b:
            return True
        
        # Подстрока
        if a in b or b in a:
            return True
            
        # Jaccard similarity для коротких строк
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return (intersection / union) > threshold if union > 0 else False

    def score_sample_advanced(self, sample, query_tags, target_tempo=120, genre_hint=None, energy_level=0.5):
        """Продвинутая система скоринга сэмплов"""
        score = 0
        sample_tags = sample.get("tags", [])
        sample_tempo = sample.get("tempo", 120)
        sample_path = sample.get("path", "").lower()
        
        # 1. Семантический скоринг тегов (основной вес)
        tag_score = 0
        for query_tag in query_tags:
            semantic_score = self.find_semantic_matches(query_tag, sample_tags)
            tag_score += semantic_score
        
        # Нормализация тегового скора
        if query_tags:
            tag_score = tag_score / len(query_tags)
        
        score += tag_score * 0.6  # 60% веса
        
        # 2. Темповая совместимость (более гибкая)
        tempo_diff = abs(sample_tempo - target_tempo)
        if tempo_diff <= 5:
            tempo_score = 20
        elif tempo_diff <= 10:
            tempo_score = 15
        elif tempo_diff <= 20:
            tempo_score = 10
        elif tempo_diff <= 40:
            tempo_score = 5
        else:
            tempo_score = 1  # Всё равно даём шанс
            
        score += tempo_score * 0.2  # 20% веса
        
        # 3. Жанровая совместимость
        genre_score = 0
        if genre_hint:
            if genre_hint in sample_path:
                genre_score = 15
            elif any(g in sample_path for g in [genre_hint[:4], genre_hint[:3]]):
                genre_score = 10
                
        score += genre_score * 0.1  # 10% веса
        
        # 4. Качество сэмпла
        quality_score = 0
        
        # Предпочитаем сэмплы с хорошими тегами
        if len(sample_tags) > 2:
            quality_score += 5
            
        # Предпочитаем сэмплы с определённой тональностью
        if sample.get("key"):
            quality_score += 3
            
        # Предпочитаем сэмплы подходящей длительности
        duration = sample.get("duration", 0)
        if 1 < duration < 60:  # От 1 секунды до минуты
            quality_score += 5
        
        score += quality_score * 0.1  # 10% веса
        
        return score

    def pick_samples_enhanced(self, required_tags, target_tempo=120, genre_hint=None, 
                            energy_level=0.5, top_k=10, min_score=5):
        """Улучшенный подбор сэмплов с множественными стратегиями"""
        
        if not self.index:
            logging.error("❌ Индекс пуст!")
            return []
        
        logging.info(f"🎯 Поиск сэмплов для тегов: {required_tags}, темп: {target_tempo}")
        
        # Стратегия 1: Точный поиск
        exact_matches = []
        for sample in self.index:
            score = self.score_sample_advanced(
                sample, required_tags, target_tempo, genre_hint, energy_level
            )
            if score >= min_score:
                exact_matches.append((score, sample))
        
        # Стратегия 2: Если мало точных совпадений, расширяем поиск
        if len(exact_matches) < top_k:
            logging.info("🔄 Расширяю критерии поиска...")
            
            # Более мягкие критерии
            for sample in self.index:
                if any((score, sample) in exact_matches for score, _ in exact_matches):
                    continue
                    
                score = self.score_sample_advanced(
                    sample, required_tags, target_tempo, genre_hint, energy_level
                )
                
                # Понижаем порог
                if score >= min_score * 0.5:
                    exact_matches.append((score, sample))
        
        # Стратегия 3: Фолбек - берём лучшие по жанру
        if len(exact_matches) < 3:
            logging.info("🆘 Фолбек: поиск по жанру и темпу...")
            
            for sample in self.index:
                sample_path = sample.get("path", "").lower()
                sample_tempo = sample.get("tempo", 120)
                
                # Жанровое совпадение или близкий темп
                genre_match = genre_hint and genre_hint in sample_path
                tempo_match = abs(sample_tempo - target_tempo) <= 30
                
                if genre_match or tempo_match:
                    fallback_score = 3 + (5 if genre_match else 0) + (2 if tempo_match else 0)
                    exact_matches.append((fallback_score, sample))
        
        # Сортировка и возврат
        exact_matches.sort(key=lambda x: -x[0])
        top_samples = [sample for _, sample in exact_matches[:top_k]]
        
        logging.info(f"✅ Найдено сэмплов: {len(top_samples)} (из {len(self.index)})")
        
        # Логирование для отладки
        if top_samples:
            best_sample = top_samples[0]
            logging.info(f"🏆 Лучший: {best_sample['filename']} | теги: {best_sample['tags']} | темп: {best_sample['tempo']}")
        
        return top_samples

class SmartMixer:
    """Улучшенная система микширования с интеллектуальным подбором"""
    
    def __init__(self, sample_picker):
        self.picker = sample_picker
    
    def create_enhanced_mix(self, json_data, output_dir="output"):
        """Создание микса с улучшенным алгоритмом"""
        os.makedirs(output_dir, exist_ok=True)
        
        tempo = json_data.get("tempo", 120)
        structure = json_data.get("structure", [])
        tracks = json_data.get("tracks", [])
        
        total_duration_sec = sum(s["duration"] for s in structure)
        total_duration_ms = int(total_duration_sec * 1000)
        
        # Создаём базовый микс
        final_mix = AudioSegment.silent(duration=total_duration_ms)
        
        # Определяем жанр из первого трека для контекста
        genre_hint = self.detect_genre_from_tracks(tracks)
        
        successful_tracks = 0
        
        for track_idx, track in enumerate(tracks):
            track_name = track.get("name", f"track_{track_idx}")
            sample_tags = track.get("sample_tags", [])
            volume = track.get("volume", -6)
            
            # Тайминг трека
            starts_at_beats = track.get("starts_at", 0)
            ends_at_beats = track.get("ends_at", None)
            
            beat_duration = 60.0 / tempo
            starts_at_ms = int(starts_at_beats * beat_duration * 1000)
            ends_at_ms = int(ends_at_beats * beat_duration * 1000) if ends_at_beats else total_duration_ms
            
            track_duration_ms = ends_at_ms - starts_at_ms
            
            logging.info(f"🎵 Обработка трека: {track_name} ({sample_tags})")
            
            # Подбор сэмплов с множественными стратегиями
            picked_samples = self.picker.pick_samples_enhanced(
                required_tags=sample_tags,
                target_tempo=tempo,
                genre_hint=genre_hint,
                top_k=5
            )
            
            if picked_samples:
                # Выбираем лучший сэмпл
                chosen_sample = picked_samples[0]
                sample_path = chosen_sample["path"]
                
                try:
                    # Загружаем и обрабатываем сэмпл
                    sample_audio = AudioSegment.from_file(sample_path)
                    
                    # Подгонка длительности
                    if len(sample_audio) > track_duration_ms:
                        # Для лупов - режем, для one-shots - используем полностью
                        if chosen_sample.get("category") == "loop":
                            sample_audio = sample_audio[:track_duration_ms]
                    elif len(sample_audio) < track_duration_ms:
                        # Зацикливаем или дополняем тишиной
                        if chosen_sample.get("category") == "loop":
                            repeats = track_duration_ms // len(sample_audio) + 1
                            sample_audio = (sample_audio * repeats)[:track_duration_ms]
                        else:
                            # One-shot + тишина
                            pad_duration = track_duration_ms - len(sample_audio)
                            sample_audio += AudioSegment.silent(duration=pad_duration)
                    
                    # Применяем volume
                    sample_audio = sample_audio + volume
                    
                    # Добавляем в микс
                    final_mix = final_mix.overlay(sample_audio, position=starts_at_ms)
                    
                    # Сохраняем stem
                    stem_path = os.path.join(output_dir, f"{track_name}.wav")
                    sample_audio.export(stem_path, format="wav")
                    
                    successful_tracks += 1
                    logging.info(f"✅ [{track_name}] добавлен: {chosen_sample['filename']}")
                    
                except Exception as e:
                    logging.error(f"❌ Ошибка обработки сэмпла {sample_path}: {e}")
            else:
                logging.warning(f"⚠️ Не найдены сэмплы для {track_name}")
        
        # Сохраняем финальный микс
        final_path = os.path.join(output_dir, "final_mix.wav")
        final_mix.export(final_path, format="wav")
        
        logging.info(f"🎛️ Микс готов: {successful_tracks}/{len(tracks)} треков | {final_path}")
        return final_mix, final_path

    def detect_genre_from_tracks(self, tracks):
        """Определение жанра из названий треков"""
        track_names = " ".join([t.get("name", "") for t in tracks]).lower()
        
        genre_keywords = {
            "trap": ["trap", "drill", "808", "dark"],
            "house": ["house", "groove", "dance"],
            "techno": ["techno", "tech", "industrial"],
            "ambient": ["ambient", "pad", "texture"],
            "dnb": ["dnb", "drum", "bass", "break"]
        }
        
        for genre, keywords in genre_keywords.items():
            if any(kw in track_names for kw in keywords):
                return genre
        
        return None

class EnhancedComposerEngine:
    """Главный движок с улучшенной логикой"""
    
    def __init__(self, sample_dir):
        self.sample_picker = EnhancedSamplePicker(sample_dir)
        self.mixer = SmartMixer(self.sample_picker)
    
    def process_prompt(self, prompt, output_dir="output"):
        """Обработка промпта с созданием трека"""
        
        # Парсинг промпта
        extracted_info = self.parse_prompt_enhanced(prompt)
        
        # Создание структуры трека
        track_structure = self.create_track_structure(extracted_info)
        
        # Сохранение JSON отчёта
        json_path = os.path.join(output_dir, "track_report.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(track_structure, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📄 Структура трека сохранена: {json_path}")
        
        # Создание микса
        final_mix, final_path = self.mixer.create_enhanced_mix(track_structure, output_dir)
        
        return final_path, track_structure

    def parse_prompt_enhanced(self, prompt):
        """Расширенный парсинг промпта"""
        prompt_lower = prompt.lower()
        
        # Извлечение BPM
        bpm_match = re.search(r'(\d{2,3})\s*bpm', prompt_lower)
        tempo = int(bpm_match.group(1)) if bpm_match else 120
        
        # Определение жанра
        genre = self.detect_genre_from_prompt(prompt_lower)
        
        # Извлечение настроения
        mood_tags = self.extract_mood_tags(prompt_lower)
        
        # Извлечение инструментов
        instrument_mentions = self.extract_instruments(prompt_lower)
        
        return {
            "tempo": tempo,
            "genre": genre,
            "mood": mood_tags,
            "instruments": instrument_mentions,
            "original_prompt": prompt
        }

    def detect_genre_from_prompt(self, prompt):
        """Определение жанра из промпта"""
        genre_patterns = {
            "trap": ["trap", "drill", "агрессивн", "тёмн", "жёстк"],
            "house": ["house", "groove", "танц", "диско"],
            "techno": ["techno", "техничн", "индастриал", "минимал"],
            "ambient": ["ambient", "атмосфер", "расслабл", "медитат"],
            "dnb": ["dnb", "drum", "bass", "энерг", "быстр"]
        }
        
        for genre, patterns in genre_patterns.items():
            if any(p in prompt for p in patterns):
                return genre
        
        return "electronic"  # дефолт

    def extract_mood_tags(self, prompt):
        """Извлечение тегов настроения"""
        mood_patterns = {
            "dark": ["мрачн", "тёмн", "злой", "агрессивн"],
            "energetic": ["энерг", "драйв", "мощн", "громк"],
            "melodic": ["мелодичн", "красив", "гармон"],
            "aggressive": ["агрессивн", "жёстк", "злой", "резк"],
            "chill": ["спокойн", "расслабл", "мягк", "тёпл"],
            "modern": ["современ", "техничн", "цифров"],
            "atmospheric": ["атмосфер", "воздушн", "пространств"]
        }
        
        found_moods = []
        for mood, patterns in mood_patterns.items():
            if any(p in prompt for p in patterns):
                found_moods.append(mood)
        
        return found_moods

    def extract_instruments(self, prompt):
        """Извлечение упоминаний инструментов"""
        instrument_patterns = {
            "vocal": ["вокал", "голос", "пени"],
            "piano": ["пиан", "клавиш"],
            "guitar": ["гитар"],
            "drums": ["барабан", "драм", "ударн"],
            "bass": ["бас", "808"],
            "synth": ["синт", "электрон"]
        }
        
        found_instruments = []
        for instrument, patterns in instrument_patterns.items():
            if any(p in prompt for p in patterns):
                found_instruments.append(instrument)
        
        return found_instruments

    def create_track_structure(self, extracted_info):
        """Создание структуры трека на основе извлечённой информации"""
        tempo = extracted_info["tempo"]
        genre = extracted_info["genre"]
        mood = extracted_info["mood"]
        instruments = extracted_info["instruments"]
        
        # Создание структуры секций
        structure = self.generate_sections_by_genre(genre, tempo)
        
        # Создание треков
        tracks = self.generate_tracks_by_genre(genre, mood, instruments, tempo)
        
        return {
            "tempo": tempo,
            "genre": genre,
            "mood": mood,
            "structure": structure,
            "tracks": tracks,
            "metadata": {
                "created_by": "WaveDream Enhanced",
                "prompt": extracted_info["original_prompt"]
            }
        }

    def generate_sections_by_genre(self, genre, tempo):
        """Генерация структуры секций по жанру"""
        if genre == "trap":
            return [
                {"type": "intro", "duration": 8, "start": 0},
                {"type": "verse", "duration": 16, "start": 8},
                {"type": "hook", "duration": 16, "start": 24},
                {"type": "verse", "duration": 16, "start": 40},
                {"type": "hook", "duration": 16, "start": 56},
                {"type": "bridge", "duration": 8, "start": 72},
                {"type": "outro", "duration": 8, "start": 80}
            ]
        elif genre == "house":
            return [
                {"type": "intro", "duration": 16, "start": 0},
                {"type": "build", "duration": 16, "start": 16},
                {"type": "drop", "duration": 32, "start": 32},
                {"type": "break", "duration": 16, "start": 64},
                {"type": "drop", "duration": 32, "start": 80},
                {"type": "outro", "duration": 16, "start": 112}
            ]
        elif genre == "ambient":
            return [
                {"type": "intro", "duration": 20, "start": 0},
                {"type": "development", "duration": 40, "start": 20},
                {"type": "climax", "duration": 30, "start": 60},
                {"type": "resolution", "duration": 30, "start": 90}
            ]
        else:  # electronic default
            return [
                {"type": "intro", "duration": 8, "start": 0},
                {"type": "build", "duration": 16, "start": 8},
                {"type": "drop", "duration": 24, "start": 24},
                {"type": "break", "duration": 8, "start": 48},
                {"type": "drop", "duration": 24, "start": 56},
                {"type": "outro", "duration": 8, "start": 80}
            ]

    def generate_tracks_by_genre(self, genre, mood, instruments, tempo):
        """Генерация треков по жанру с учётом настроения"""
        tracks = []
        
        # Базовые треки по жанру
        if genre == "trap":
            base_tracks = [
                {"name": "Kick", "sample_tags": ["kick", "808", "sub"], "volume": -3, "starts_at": 0},
                {"name": "Snare", "sample_tags": ["snare", "clap", "crack"], "volume": -6, "starts_at": 8},
                {"name": "HiHats", "sample_tags": ["hihat", "hat", "cymbal"], "volume": -9, "starts_at": 4},
                {"name": "Bass", "sample_tags": ["bass", "808", "sub", "reese"], "volume": -6, "starts_at": 8},
            ]
            
            # Добавляем мелодические элементы если нужно
            if "melodic" in mood or "piano" in instruments:
                base_tracks.append({
                    "name": "Lead", "sample_tags": ["lead", "melody", "synth", "piano"], 
                    "volume": -9, "starts_at": 16
                })
                
            # Добавляем вокал если упомянут
            if "vocal" in instruments:
                base_tracks.append({
                    "name": "Vocal", "sample_tags": ["vocal", "voice", "vox", "rap"], 
                    "volume": -6, "starts_at": 24
                })
                
        elif genre == "house":
            base_tracks = [
                {"name": "Kick", "sample_tags": ["kick", "four_on_floor"], "volume": -3, "starts_at": 0},
                {"name": "HiHats", "sample_tags": ["hihat", "groove", "swing"], "volume": -9, "starts_at": 0},
                {"name": "Bass", "sample_tags": ["bass", "groove", "funky"], "volume": -6, "starts_at": 16},
                {"name": "Lead", "sample_tags": ["lead", "piano", "organ", "synth"], "volume": -6, "starts_at": 32}
            ]
            
        elif genre == "ambient":
            base_tracks = [
                {"name": "Pad", "sample_tags": ["pad", "ambient", "texture"], "volume": -9, "starts_at": 0},
                {"name": "Texture", "sample_tags": ["fx", "ambient", "drone"], "volume": -12, "starts_at": 10},
                {"name": "Melody", "sample_tags": ["piano", "bell", "soft"], "volume": -9, "starts_at": 20}
            ]
            
        else:  # electronic default
            base_tracks = [
                {"name": "Drums", "sample_tags": ["kick", "snare", "hihat"], "volume": -3, "starts_at": 0},
                {"name": "Bass", "sample_tags": ["bass", "sub"], "volume": -6, "starts_at": 8},
                {"name": "Lead", "sample_tags": ["lead", "synth", "melody"], "volume": -6, "starts_at": 16}
            ]
        
        # Модификация по настроению
        if "dark" in mood:
            for track in base_tracks:
                track["sample_tags"].extend(["dark", "minor", "aggressive"])
        
        if "energetic" in mood:
            for track in base_tracks:
                track["sample_tags"].extend(["hard", "punchy", "loud"])
                track["volume"] += 2  # Громче
        
        return base_tracks

# Утилиты для работы с проектом
class ProjectManager:
    """Менеджер проектов для организации результатов"""
    
    @staticmethod
    def create_project_structure(output_dir):
        """Создание структуры папок проекта"""
        folders = ["stems", "midi", "samples_used", "reports"]
        for folder in folders:
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    @staticmethod
    def generate_project_report(track_data, output_path):
        """Генерация подробного отчёта о проекте"""
        report = {
            "project_info": {
                "tempo": track_data["tempo"],
                "genre": track_data["genre"],
                "total_duration": sum(s["duration"] for s in track_data["structure"]),
                "tracks_count": len(track_data["tracks"])
            },
            "structure_breakdown": track_data["structure"],
            "tracks_breakdown": track_data["tracks"],
            "generation_stats": {
                "successful_tracks": len([t for t in track_data["tracks"] if t.get("sample_tags")]),
                "fallback_tracks": 0  # Будет обновлено при миксе
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

# Главная функция запуска
def main():
    """Главная функция с улучшенным интерфейсом"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🎼 WaveDream Enhanced - Улучшенная генерация музыки")
    parser.add_argument("--prompt", type=str, required=True, help="Промпт для генерации")
    parser.add_argument("--sample-dir", type=str, required=True, help="Директория с сэмплами")
    parser.add_argument("--output-dir", type=str, default="output_enhanced", help="Выходная директория")
    parser.add_argument("--rebuild-index", action="store_true", help="Пересоздать индекс сэмплов")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Инициализация движка
    engine = EnhancedComposerEngine(args.sample_dir)
    
    # Пересоздание индекса если нужно
    if args.rebuild_index:
        logging.info("🔄 Пересоздание индекса...")
        engine.sample_picker.index = engine.sample_picker.build_enhanced_index()
        engine.sample_picker.save_index(engine.sample_picker.index)
    
    # Создание структуры проекта
    ProjectManager.create_project_structure(args.output_dir)
    
    # Генерация трека
    logging.info(f"🚀 Начинаю генерацию: '{args.prompt}'")
    
    try:
        final_path, track_data = engine.process_prompt(args.prompt, args.output_dir)
        
        # Создание отчёта
        report_path = os.path.join(args.output_dir, "reports", "project_report.json")
        ProjectManager.generate_project_report(track_data, report_path)
        
        logging.info(f"🎉 Генерация завершена!")
        logging.info(f"📁 Финальный трек: {final_path}")
        logging.info(f"📊 Отчёт: {report_path}")
        
        # Статистика
        total_samples = len(engine.sample_picker.index)
        used_samples = len([t for t in track_data["tracks"] if t.get("sample_tags")])
        
        print(f"\n{'='*50}")
        print(f"🎵 РЕЗУЛЬТАТ ГЕНЕРАЦИИ")
        print(f"{'='*50}")
        print(f"Жанр: {track_data['genre']}")
        print(f"Темп: {track_data['tempo']} BPM")
        print(f"Треков: {len(track_data['tracks'])}")
        print(f"Длительность: {sum(s['duration'] for s in track_data['structure'])} сек")
        print(f"Использовано сэмплов: {used_samples}")
        print(f"Всего в базе: {total_samples}")
        print(f"Выходной файл: {final_path}")
        print(f"{'='*50}")
        
    except Exception as e:
        logging.error(f"❌ Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Дополнительные утилиты для тестирования и отладки

class DebugTools:
    """Инструменты для отладки и тестирования системы"""
    
    @staticmethod
    def test_sample_picker(sample_dir, test_tags=None):
        """Тестирование подбора сэмплов"""
        picker = EnhancedSamplePicker(sample_dir)
        
        test_cases = test_tags or [
            ["kick", "trap"],
            ["bass", "808"],
            ["lead", "synth"],
            ["vocal", "rap"],
            ["hihat", "trap"],
            ["piano", "melodic"]
        ]
        
        print("\n🧪 ТЕСТИРОВАНИЕ ПОДБОРА СЭМПЛОВ")
        print("="*60)
        
        for tags in test_cases:
            print(f"\n🎯 Тест тегов: {tags}")
            results = picker.pick_samples_enhanced(tags, target_tempo=140, top_k=3)
            
            if results:
                for i, sample in enumerate(results, 1):
                    print(f"  {i}. {sample['filename']}")
                    print(f"     Теги: {sample['tags']}")
                    print(f"     Темп: {sample['tempo']} BPM")
            else:
                print("  ❌ Совпадений не найдено")
    
    @staticmethod
    def analyze_index_quality(sample_dir):
        """Анализ качества индексации"""
        picker = EnhancedSamplePicker(sample_dir)
        index = picker.index
        
        # Статистика тегов
        all_tags = []
        empty_tags = 0
        tempo_distribution = []
        
        for sample in index:
            tags = sample.get("tags", [])
            if not tags:
                empty_tags += 1
            else:
                all_tags.extend(tags)
            
            tempo_distribution.append(sample.get("tempo", 120))
        
        tag_counter = Counter(all_tags)
        
        print("\n📊 АНАЛИЗ КАЧЕСТВА ИНДЕКСА")
        print("="*60)
        print(f"Всего сэмплов: {len(index)}")
        print(f"Без тегов: {empty_tags} ({empty_tags/len(index)*100:.1f}%)")
        print(f"Уникальных тегов: {len(tag_counter)}")
        print(f"\n🏷️ Топ-10 тегов:")
        
        for tag, count in tag_counter.most_common(10):
            print(f"  {tag}: {count}")
        
        print(f"\n🎵 Распределение темпа:")
        tempo_ranges = {
            "Slow (60-90)": len([t for t in tempo_distribution if 60 <= t < 90]),
            "Medium (90-130)": len([t for t in tempo_distribution if 90 <= t < 130]),
            "Fast (130-180)": len([t for t in tempo_distribution if 130 <= t < 180]),
            "Very Fast (180+)": len([t for t in tempo_distribution if t >= 180])
        }
        
        for range_name, count in tempo_ranges.items():
            print(f"  {range_name}: {count}")

# Конфигурация для quick start
QUICK_START_CONFIG = {
    "sample_directories": {
        "default": "D:\\0\\шаблоны\\Samples for AKAI",
        "backup": "samples"  # fallback директория
    },
    "output_directory": "wavedream_output",
    "test_prompts": [
        "trap с вокалом 160bpm мрачный агрессивный",
        "house groove 125bpm melodic danceable", 
        "ambient atmospheric 80bpm relaxing chill",
        "techno industrial 130bpm hard driving"
    ]
}

def quick_test():
    """Быстрый тест системы"""
    config = QUICK_START_CONFIG
    sample_dir = config["sample_directories"]["default"]
    
    if not os.path.exists(sample_dir):
        sample_dir = config["sample_directories"]["backup"]
        if not os.path.exists(sample_dir):
            print("❌ Не найдена директория с сэмплами")
            return
    
    print("🚀 Быстрый тест WaveDream Enhanced")
    print(f"📂 Используем сэмплы из: {sample_dir}")
    
    # Тестируем подбор сэмплов
    DebugTools.test_sample_picker(sample_dir)
    
    # Анализируем качество индекса
    DebugTools.analyze_index_quality(sample_dir)
    
    print("\n✅ Быстрый тест завершён")

# Запуск quick test при прямом вызове модуля
if __name__ == "__main__" and len(os.sys.argv) == 1:
    quick_test()