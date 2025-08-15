# metadata.py - Процессор метаданных и анализа промптов

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import json

from config import config, GenreType


class MetadataProcessor:
    """
    Процессор метаданных и анализа промптов
    
    Возможности:
    - Глубокий анализ текстовых промптов
    - Извлечение музыкальных параметров (BPM, тональность, настроение)
    - Детекция жанров по ключевым словам
    - Семантический анализ инструментов
    - Извлечение структурных подсказок
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Компиляция регулярных выражений для производительности
        self.regex_patterns = {
            'bpm': re.compile(r'(\d{2,3})\s*(?:bpm|beats?|tempo)', re.IGNORECASE),
            'key': re.compile(r'\b([A-G][#b]?)\s*(?:maj|min|major|minor)?\b', re.IGNORECASE),
            'duration': re.compile(r'(\d+)\s*(?:min|minutes?|sec|seconds?)', re.IGNORECASE),
            'energy': re.compile(r'\b(high|low|medium|intense|calm|aggressive|peaceful)\s*energy\b', re.IGNORECASE)
        }
        
        # Словари для семантического анализа
        self.mood_keywords = {
            'dark': ['dark', 'evil', 'sinister', 'goth', 'horror', 'mysterious', 'shadow'],
            'bright': ['bright', 'happy', 'joyful', 'uplifting', 'cheerful', 'sunny', 'positive'],
            'aggressive': ['aggressive', 'hard', 'intense', 'brutal', 'heavy', 'powerful', 'strong'],
            'calm': ['calm', 'peaceful', 'soft', 'gentle', 'relaxing', 'chill', 'smooth'],
            'energetic': ['energetic', 'driving', 'pumping', 'dynamic', 'active', 'bouncy'],
            'melancholic': ['sad', 'melancholic', 'emotional', 'nostalgic', 'wistful', 'longing'],
            'ethereal': ['ethereal', 'ambient', 'spacious', 'floating', 'dreamy', 'atmospheric'],
            'vintage': ['vintage', 'retro', 'old-school', 'classic', 'nostalgic', 'analog']
        }
        
        self.instrument_aliases = {
            'drums': ['drums', 'percussion', 'kit', 'beats'],
            'bass': ['bass', 'sub', '808', 'low-end', 'bottom'],
            'synth': ['synth', 'synthesizer', 'keys', 'keyboard', 'lead'],
            'guitar': ['guitar', 'gtr', 'strings', 'riff'],
            'vocal': ['vocal', 'voice', 'singing', 'rap', 'lyrics'],
            'piano': ['piano', 'keys', 'ivories'],
            'strings': ['strings', 'orchestra', 'violin', 'cello'],
            'brass': ['brass', 'trumpet', 'horn', 'trombone']
        }
        
        # Жанровые индикаторы из config
        self.genre_keywords = {}
        for genre_enum in GenreType:
            genre = genre_enum.value
            if genre in config.GENRE_CONFIGS:
                genre_config = config.GENRE_CONFIGS[genre] 
                self.genre_keywords[genre] = getattr(genre_config, 'default_tags', [])
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Полный анализ текстового промпта
        
        Args:
            prompt: Исходный текст промпта
            
        Returns:
            Словарь с извлечёнными параметрами и метаданными
        """
        self.logger.info(f"📝 Analyzing prompt: '{prompt[:50]}...'")
        
        analysis = {
            'original_prompt': prompt,
            'cleaned_prompt': self._clean_prompt(prompt),
            'language': self._detect_language(prompt),
            'length': len(prompt),
            'word_count': len(prompt.split())
        }
        
        # Извлекаем различные параметры
        analysis.update(self.extract_parameters(prompt))
        analysis.update(self._extract_mood_descriptors(prompt))
        analysis.update(self._extract_structure_hints(prompt))
        analysis.update(self._extract_production_hints(prompt))
        
        # Анализируем сложность промпта
        analysis['complexity_score'] = self._calculate_prompt_complexity(analysis)
        
        self.logger.info(f"  📊 Extracted: {len(analysis)} metadata fields")
        
        return analysis
    
    def extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Извлечение конкретных музыкальных параметров"""
        prompt_lower = prompt.lower()
        extracted = {}
        
        # BPM извлечение
        bpm_match = self.regex_patterns['bpm'].search(prompt)
        if bpm_match:
            bpm = int(bpm_match.group(1))
            if 60 <= bpm <= 200:  # Валидный диапазон
                extracted['bpm'] = bpm
        
        # Тональность
        key_match = self.regex_patterns['key'].search(prompt)
        if key_match:
            key = key_match.group(1).upper()
            # Добавляем определение мажор/минор если есть
            full_match = key_match.group(0).lower()
            if 'min' in full_match:
                key += 'm'
            elif 'maj' in full_match:
                key += 'M'
            extracted['key'] = key
        
        # Длительность
        duration_match = self.regex_patterns['duration'].search(prompt)
        if duration_match:
            duration_value = int(duration_match.group(1))
            duration_unit = duration_match.group(0).lower()
            
            if 'min' in duration_unit:
                extracted['duration'] = duration_value * 60
            else:
                extracted['duration'] = duration_value
        
        # Уровень энергии
        energy_match = self.regex_patterns['energy'].search(prompt)
        if energy_match:
            energy_word = energy_match.group(1).lower()
            energy_mapping = {
                'low': 0.3, 'calm': 0.3, 'peaceful': 0.2,
                'medium': 0.5,
                'high': 0.8, 'intense': 0.9, 'aggressive': 0.9
            }
            extracted['energy_level'] = energy_mapping.get(energy_word, 0.5)
        
        # Инструменты
        detected_instruments = []
        for instrument, aliases in self.instrument_aliases.items():
            for alias in aliases:
                if alias in prompt_lower:
                    detected_instruments.append(instrument)
                    break
        
        if detected_instruments:
            extracted['instruments'] = list(set(detected_instruments))
        
        # Теги из семантической карты
        detected_tags = []
        for tag, data in config.SEMANTIC_MAP.items():
            if isinstance(data, dict):
                synonyms = data.get('synonyms', [])
            else:
                synonyms = data if isinstance(data, list) else [data]
            
            for synonym in synonyms:
                if synonym.lower() in prompt_lower:
                    detected_tags.append(tag)
                    break
        
        if detected_tags:
            extracted['tags'] = list(set(detected_tags))
        
        return extracted
    
    def detect_genre(self, prompt: str, existing_tags: Optional[List[str]] = None) -> str:
        """
        Детекция жанра на основе промпта и тегов
        
        Args:
            prompt: Текст промпта
            existing_tags: Уже извлечённые теги
            
        Returns:
            Название детектированного жанра
        """
        prompt_lower = prompt.lower()
        genre_scores = {}
        
        # 1. Прямое совпадение названий жанров
        for genre in self.genre_keywords:
            if genre in prompt_lower:
                genre_scores[genre] = genre_scores.get(genre, 0) + 10
        
        # 2. Совпадение по ключевым словам
        for genre, keywords in self.genre_keywords.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    genre_scores[genre] = genre_scores.get(genre, 0) + 3
        
        # 3. Анализ существующих тегов
        if existing_tags:
            for tag in existing_tags:
                tag_lower = tag.lower()
                for genre, keywords in self.genre_keywords.items():
                    if tag_lower in [k.lower() for k in keywords]:
                        genre_scores[genre] = genre_scores.get(genre, 0) + 5
        
        # 4. Контекстный анализ
        context_indicators = {
            'trap': ['urban', 'street', 'hip-hop', 'drill', 'memphis'],
            'lofi': ['study', 'chill', 'vintage', 'vinyl', 'cozy'],
            'dnb': ['jungle', 'breakbeat', 'neurofunk', 'liquid'],
            'ambient': ['meditation', 'space', 'atmospheric', 'zen'],
            'techno': ['warehouse', 'industrial', 'minimal', 'berlin'],
            'house': ['disco', 'dance', 'club', 'groove', 'funk']
        }
        
        for genre, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in prompt_lower:
                    genre_scores[genre] = genre_scores.get(genre, 0) + 2
        
        # Выбираем жанр с наибольшим скором
        if genre_scores:
            best_genre = max(genre_scores, key=genre_scores.get)
            confidence = genre_scores[best_genre]
            
            self.logger.info(f"  🎭 Genre detected: {best_genre} (confidence: {confidence})")
            return best_genre
        
        # Фолбек - возвращаем trap как наиболее универсальный
        self.logger.info("  🎭 Genre detection fallback: trap")
        return "trap"
    
    def _clean_prompt(self, prompt: str) -> str:
        """Очистка и нормализация промпта"""
        # Убираем лишние пробелы и символы
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        
        # Убираем специальные символы, оставляем только буквы, цифры, пробелы и основную пунктуацию
        cleaned = re.sub(r'[^\w\s\-.,!?#]', '', cleaned)
        
        return cleaned
    
    def _detect_language(self, prompt: str) -> str:
        """Простая детекция языка"""
        # Проверяем наличие кириллицы
        if re.search(r'[а-яё]', prompt.lower()):
            return 'ru'
        # Проверяем базовые английские слова
        elif re.search(r'\b(and|the|with|for|of|in|to|a|an)\b', prompt.lower()):
            return 'en'
        else:
            return 'unknown'
    
    def _extract_mood_descriptors(self, prompt: str) -> Dict[str, Any]:
        """Извлечение дескрипторов настроения"""
        prompt_lower = prompt.lower()
        detected_moods = []
        mood_confidence = {}
        
        for mood, keywords in self.mood_keywords.items():
            matches = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    matches += 1
            
            if matches > 0:
                detected_moods.append(mood)
                mood_confidence[mood] = matches / len(keywords)
        
        result = {'mood': detected_moods}
        
        if mood_confidence:
            # Определяем доминирующее настроение
            dominant_mood = max(mood_confidence, key=mood_confidence.get)
            result['dominant_mood'] = dominant_mood
            result['mood_confidence'] = mood_confidence
        
        return result
    
    def _extract_structure_hints(self, prompt: str) -> Dict[str, Any]:
        """Извлечение подсказок о структуре"""
        prompt_lower = prompt.lower()
        structure_hints = {}
        
        # Поиск упоминаний структурных элементов
        structure_elements = {
            'intro': ['intro', 'introduction', 'start', 'beginning'],
            'verse': ['verse', 'vocal', 'rap'],
            'chorus': ['chorus', 'hook', 'refrain'],
            'bridge': ['bridge', 'breakdown', 'middle'],
            'outro': ['outro', 'end', 'ending', 'fade'],
            'drop': ['drop', 'climax', 'peak'],
            'build': ['build', 'tension', 'rising']
        }
        
        found_elements = []
        for element, keywords in structure_elements.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    found_elements.append(element)
                    break
        
        if found_elements:
            structure_hints['mentioned_sections'] = list(set(found_elements))
        
        # Поиск указаний на длительность секций
        section_durations = re.findall(r'(\w+)\s+(?:for\s+)?(\d+)\s*(?:sec|bar|beat)', prompt_lower)
        if section_durations:
            structure_hints['section_durations'] = dict(section_durations)
        
        return structure_hints
    
    def _extract_production_hints(self, prompt: str) -> Dict[str, Any]:
        """Извлечение подсказок о продакшене"""
        prompt_lower = prompt.lower()
        production_hints = {}
        
        # Обработка и эффекты
        effects_keywords = {
            'reverb': ['reverb', 'echo', 'hall', 'space'],
            'delay': ['delay', 'echo', 'repeat'],
            'distortion': ['distortion', 'overdrive', 'saturated', 'dirty'],
            'compression': ['compressed', 'punchy', 'tight'],
            'filtering': ['filtered', 'muffled', 'bright', 'dark']
        }
        
        mentioned_effects = []
        for effect, keywords in effects_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    mentioned_effects.append(effect)
                    break
        
        if mentioned_effects:
            production_hints['suggested_effects'] = mentioned_effects
        
        # Качественные характеристики
        quality_keywords = {
            'professional': ['professional', 'studio', 'mastered', 'polished'],
            'lofi': ['lofi', 'vintage', 'analog', 'warm', 'tape'],
            'clean': ['clean', 'crisp', 'clear', 'pristine'],
            'raw': ['raw', 'unprocessed', 'live', 'organic']
        }
        
        quality_indicators = []
        for quality, keywords in quality_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    quality_indicators.append(quality)
                    break
        
        if quality_indicators:
            production_hints['quality_style'] = quality_indicators
        
        return production_hints
    
    def _calculate_prompt_complexity(self, analysis: Dict) -> float:
        """Расчёт сложности промпта для адаптации алгоритмов"""
        complexity_score = 0.0
        
        # Базовая сложность по количеству слов
        word_count = analysis.get('word_count', 0)
        complexity_score += min(1.0, word_count / 20)  # Нормализация
        
        # Бонус за конкретные параметры
        if 'bpm' in analysis:
            complexity_score += 0.2
        if 'key' in analysis:
            complexity_score += 0.2
        if 'duration' in analysis:
            complexity_score += 0.1
        
        # Бонус за инструментальные указания
        instruments_count = len(analysis.get('instruments', []))
        complexity_score += min(0.3, instruments_count * 0.1)
        
        # Бонус за настроения
        moods_count = len(analysis.get('mood', []))
        complexity_score += min(0.2, moods_count * 0.05)
        
        # Бонус за структурные подсказки
        if 'mentioned_sections' in analysis:
            complexity_score += 0.2
        
        # Бонус за продакшен подсказки
        if 'suggested_effects' in analysis:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def create_generation_context(self, analysis: Dict) -> Dict[str, Any]:
        """Создание контекста для генерации на основе анализа"""
        context = {
            'prompt_analysis': analysis,
            'generation_hints': {},
            'quality_targets': {},
            'creative_constraints': {}
        }
        
        # Генерационные подсказки
        if analysis.get('complexity_score', 0) > 0.7:
            context['generation_hints']['approach'] = 'detailed'
            context['generation_hints']['creativity'] = 0.8
        else:
            context['generation_hints']['approach'] = 'standard'
            context['generation_hints']['creativity'] = 0.6
        
        # Целевые показатели качества
        if 'professional' in analysis.get('quality_style', []):
            context['quality_targets']['mastering_intensity'] = 0.8
            context['quality_targets']['dynamics_preservation'] = 0.7
        elif 'lofi' in analysis.get('quality_style', []):
            context['quality_targets']['vintage_processing'] = 0.8
            context['quality_targets']['dynamics_preservation'] = 0.4
        
        # Креативные ограничения
        mentioned_sections = analysis.get('mentioned_sections', [])
        if mentioned_sections:
            context['creative_constraints']['required_sections'] = mentioned_sections
        
        if 'bpm' in analysis:
            context['creative_constraints']['tempo_range'] = (
                analysis['bpm'] - 5, analysis['bpm'] + 5
            )
        
        return context
                
def write_report(self, f, samples, mastering, exported_files):
    # === Использованные сэмплы ===
    if samples:
        f.write(f"## 🎛️ Used Samples ({len(samples)})\n\n")
        by_instrument = {}
        for sample in samples:
            instrument = sample.get("instrument_role", "unknown")
            by_instrument.setdefault(instrument, []).append(sample)
        
        for instrument, instrument_samples in by_instrument.items():
            f.write(f"### {instrument.title()}\n")
            for sample in instrument_samples:
                filename = sample.get("filename", "unknown")
                section = sample.get("section", "unknown")
                f.write(f"- **{filename}** (in {section})\n")
            f.write("\n")

    # === Мастеринговая конфигурация ===
    f.write("## 🎚️ Mastering Configuration\n\n")
    if mastering:
        f.write(f"**Target LUFS**: {mastering.get('target_lufs', 'N/A')}\n\n")
        f.write(f"**Peak Ceiling**: {mastering.get('peak_ceiling', 'N/A')} dB\n\n")
        f.write(f"**Character**: {mastering.get('character', 'N/A')}\n\n")
        if "applied_stages" in mastering:
            f.write("**Applied Processing**:\n")
            for stage in mastering["applied_stages"]:
                f.write(f"- {stage.replace('_', ' ').title()}\n")
            f.write("\n")
    
    # === Экспортированные файлы ===
    f.write("## 📁 Exported Files\n\n")
    for file_type, file_path in exported_files.items():
        relative_path = Path(file_path).name
        f.write(f"- **{file_type.replace('_', ' ').title()}**: `{relative_path}`\n")
    f.write("\n")

def analyze_audio(self, audio, sample_rate):
    try:
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        mono_samples = samples.mean(axis=1) if audio.channels == 2 else samples

        analysis = {
            'peak': audio.max_dBFS,
            'rms': audio.rms,
            'duration': len(audio) / 1000.0
        }

        # Спектральные признаки
        spectral_centroid = librosa.feature.spectral_centroid(y=mono_samples, sr=sample_rate)
        analysis['spectral_centroid'] = float(spectral_centroid.mean())

        spectral_rolloff = librosa.feature.spectral_rolloff(y=mono_samples, sr=sample_rate)
        analysis['spectral_rolloff'] = float(spectral_rolloff.mean())

        # Стерео анализ
        if audio.channels == 2:
            correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
            analysis['stereo_correlation'] = float(correlation)
            analysis['stereo_width'] = 1.0 - abs(correlation)
        else:
            analysis['stereo_correlation'] = 1.0
            analysis['stereo_width'] = 0.0

        # Частотный анализ
        fft = np.fft.rfft(mono_samples)
        freqs = np.fft.rfftfreq(len(mono_samples), 1/sample_rate)
        magnitude = np.abs(fft)

        total_energy = np.sum(magnitude**2)
        analysis['frequency_balance'] = {
            'low': float(np.sum(magnitude[freqs < 300]**2) / total_energy),
            'mid': float(np.sum(magnitude[(freqs >= 300) & (freqs < 3000)]**2) / total_energy),
            'high': float(np.sum(magnitude[freqs >= 3000]**2) / total_energy)
        }

        # Транзиенты
        onset_frames = librosa.onset.onset_detect(y=mono_samples, sr=sample_rate)
        analysis['transient_density'] = len(onset_frames) / analysis['duration']

        # Гармоничность
        harmonic, percussive = librosa.effects.hpss(mono_samples)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        total = harmonic_energy + percussive_energy

        analysis['harmonic_ratio'] = float(harmonic_energy / total) if total > 0 else 0.5
        analysis['percussive_ratio'] = float(percussive_energy / total) if total > 0 else 0.5

        return analysis

    except Exception as e:
        self.logger.error(f"Source analysis error: {e}")
        return {
            'peak': audio.max_dBFS,
            'rms': audio.rms,
            'lufs': -20,
            'dynamic_range': 10,
            'duration': len(audio) / 1000.0,
            'frequency_balance': {'low': 0.33, 'mid': 0.33, 'high': 0.34},
            'stereo_width': 0.5,
            'harmonic_ratio': 0.5
        }

    def _create_mastering_target(self, target_config: Dict, genre_info: Dict, source_analysis: Dict) -> MasteringTarget:
        """Создание целевых параметров для мастеринга"""
        
        # Базовые цели из конфига
        target = MasteringTarget(
            lufs=target_config.get("target_lufs", -14),
            peak_ceiling=target_config.get("peak_ceiling", -1),
            dynamic_range=target_config.get("dynamic_range", 8),
            stereo_width=target_config.get("stereo_enhancement", 1.0),
            frequency_balance={},
            harmonic_content=target_config.get("harmonic_saturation", 0.2),
            transient_preservation=0.8  # По умолчанию сохраняем большинство транзиентов
        )
        
        # Адаптация под жанр
        genre = genre_info.get("name", "")
        genre_adaptations = {
            "trap": {
                "frequency_balance": {"low": 0.4, "mid": 0.35, "high": 0.25},
                "stereo_width": 1.2,
                "transient_preservation": 0.9  # Сохраняем удары
            },
            "lofi": {
                "frequency_balance": {"low": 0.35, "mid": 0.45, "high": 0.2},
                "stereo_width": 0.8,  # Более узкий стерео
                "transient_preservation": 0.6  # Более мягко
            },
            "dnb": {
                "frequency_balance": {"low": 0.3, "mid": 0.4, "high": 0.3},
                "stereo_width": 1.1,
                "transient_preservation": 0.95  # Максимум транзиентов
            },
            "ambient": {
                "frequency_balance": {"low": 0.25, "mid": 0.45, "high": 0.3},
                "stereo_width": 1.5,  # Широкий стерео
                "transient_preservation": 0.5  # Мягкость
            },
            "techno": {
                "frequency_balance": {"low": 0.35, "mid": 0.4, "high": 0.25},
                "stereo_width": 1.0,
                "transient_preservation": 0.85
            }
        }
        
        if genre in genre_adaptations:
            adaptations = genre_adaptations[genre]
            target.frequency_balance = adaptations["frequency_balance"]
            target.stereo_width = adaptations["stereo_width"]
            target.transient_preservation = adaptations["transient_preservation"]
        else:
            # Дефолтный баланс
            target.frequency_balance = {"low": 0.33, "mid": 0.34, "high": 0.33}
        
        return target
    
    async def _plan_mastering_chain(
        self, source_analysis: Dict, target: MasteringTarget, genre_info: Dict
    ) -> Dict:
        """Планирование цепочки обработки на основе анализа"""
        
        plan = {
            "stages": [],
            "parameters": {},
            "genre_specific": {}
        }
        
        # 1. Анализ необходимых коррекций
        lufs_diff = target.lufs - source_analysis["lufs"]
        peak_diff = target.peak_ceiling - source_analysis["peak"]
        
        # 2. EQ планирование
        source_balance = source_analysis.get("frequency_balance", {"low": 0.33, "mid": 0.33, "high": 0.34})
        target_balance = target.frequency_balance
        
        eq_corrections = {}
        for band in ["low", "mid", "high"]:
            source_level = source_balance.get(band, 0.33)
            target_level = target_balance.get(band, 0.33)
            
            # Конвертируем в dB коррекции (приближенно)
            if target_level > source_level:
                correction = min(6, (target_level - source_level) * 20)
            else:
                correction = max(-6, (target_level - source_level) * 20)
            
            eq_corrections[band] = correction
        
        if any(abs(c) > 0.5 for c in eq_corrections.values()):
            plan["stages"].append("eq")
            plan["parameters"]["eq"] = eq_corrections
        
        # 3. Компрессия планирование
        source_dr = source_analysis.get("dynamic_range", 10)
        target_dr = target.dynamic_range
        
        if source_dr > target_dr + 2:  # Нужна компрессия
            compression_ratio = min(8.0, 1 + (source_dr - target_dr) / 5)
            threshold = source_analysis["peak"] - (source_dr * 0.7)
            
            plan["stages"].append("compressor")
            plan["parameters"]["compressor"] = {
                "ratio": compression_ratio,
                "threshold": threshold,
                "attack": 10,  # ms
                "release": 100  # ms
            }
        
        # 4. Насыщение планирование
        if target.harmonic_content > 0.1:
            genre = genre_info.get("name", "")
            saturation_types = {
                "lofi": "tape",
                "trap": "tube", 
                "techno": "transistor",
                "ambient": "tube"
            }
            
            plan["stages"].append("saturation")
            plan["parameters"]["saturation"] = {
                "amount": target.harmonic_content,
                "type": saturation_types.get(genre, "tube"),
                "warmth": target.harmonic_content * 1.5
            }
        
        # 5. Стерео обработка
        source_width = source_analysis.get("stereo_width", 0.5)
        if abs(target.stereo_width - source_width) > 0.1:
            plan["stages"].append("stereo")
            plan["parameters"]["stereo"] = {
                "width": target.stereo_width,
                "imaging": "enhanced" if target.stereo_width > 1.0 else "natural"
            }
        
        # 6. Реверб (если требуется для жанра)
        genre = genre_info.get("name", "")
        if genre in ["ambient", "cinematic"]:
            plan["stages"].append("reverb")
            reverb_settings = {
                "ambient": {"room_size": 0.7, "wet_level": 0.25, "type": "hall"},
                "cinematic": {"room_size": 0.6, "wet_level": 0.2, "type": "cinematic_hall"}
            }
            plan["parameters"]["reverb"] = reverb_settings.get(genre, {"room_size": 0.3, "wet_level": 0.15})
        
        # 7. Лимитер (всегда последний)
        plan["stages"].append("limiter")
        plan["parameters"]["limiter"] = {
            "threshold": target.peak_ceiling + 1,  # Немного запаса
            "ceiling": target.peak_ceiling,
            "release": 50  # ms
        }
        
        # 8. Жанро-специфичные настройки
        plan["genre_specific"] = {
            "preserve_transients": target.transient_preservation,
            "target_loudness": target.lufs,
            "processing_intensity": min(1.0, abs(lufs_diff) / 10)  # Интенсивность обработки
        }
        
        self.logger.info(f"  📋 Processing plan: {len(plan['stages'])} stages - {', '.join(plan['stages'])}")
        
        return plan
    
    async def _apply_mastering_chain(self, audio: AudioSegment, plan: Dict) -> AudioSegment:
        """Применение запланированной цепочки обработки"""
        
        processed = audio
        
        for stage in plan["stages"]:
            if stage in plan["parameters"]:
                params = plan["parameters"][stage]
                
                self.logger.debug(f"  🔧 Applying {stage}: {params}")
                
                try:
                    processed = await self.effects_chain.processors[stage].process(processed, params)
                except Exception as e:
                    self.logger.error(f"❌ Error in {stage}: {e}")
        
        return processed
    
    async def _final_verification_pass(self, audio: AudioSegment, target: MasteringTarget) -> AudioSegment:
        """Финальная верификация и коррекция"""
        
        # Анализируем результат
        final_analysis = await self._analyze_source_material(audio)
        
        corrections_needed = []
        
        # Проверяем LUFS
        lufs_error = abs(final_analysis["lufs"] - target.lufs)
        if lufs_error > 1.0:  # Больше 1 LU ошибки
            corrections_needed.append("loudness")
        
        # Проверяем пики
        if final_analysis["peak"] > target.peak_ceiling:
            corrections_needed.append("peaks")
        
        # Применяем коррекции если нужно
        corrected = audio
        
        if "peaks" in corrections_needed:
            # Финальное лимитирование
            over_ceiling = final_analysis["peak"] - target.peak_ceiling
            corrected = corrected - (over_ceiling + 0.1)  # Небольшой запас
            self.logger.info(f"  🚧 Final peak correction: -{over_ceiling + 0.1:.1f}dB")
        
        if "loudness" in corrections_needed:
            # Коррекция громкости
            lufs_correction = target.lufs - final_analysis["lufs"]
            if abs(lufs_correction) < 6:  # Разумный лимит
                corrected = corrected + lufs_correction
                self.logger.info(f"  📊 Final loudness correction: {lufs_correction:+.1f}dB")
        
        return corrected
    
    def _create_mastering_report(
        self, plan: Dict, source_analysis: Dict, target: MasteringTarget
    ) -> Dict:
        """Создание отчёта о применённой обработке"""
        
        report = {
            "applied_stages": plan["stages"],
            "parameters": plan["parameters"],
            "source_characteristics": {
                "lufs": source_analysis["lufs"],
                "peak": source_analysis["peak"],
                "dynamic_range": source_analysis["dynamic_range"],
                "stereo_width": source_analysis.get("stereo_width", 0.5)
            },
            "target_characteristics": {
                "lufs": target.lufs,
                "peak_ceiling": target.peak_ceiling,
                "dynamic_range": target.dynamic_range,
                "stereo_width": target.stereo_width
            },
            "processing_intensity": plan["genre_specific"]["processing_intensity"],
            "character": f"mastered for {target.lufs} LUFS with {target.dynamic_range} LU dynamics"
        }
        
        return report