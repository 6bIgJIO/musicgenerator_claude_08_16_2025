# mix.py - Продвинутая система микширования
import math
import os
import logging
import random
import numpy as np
from pydub import AudioSegment, effects
from pydub.generators import Sine, WhiteNoise
import librosa
import soundfile as sf
from config import Config

class AdvancedMixer:
    """Продвинутая система микширования с эффектами и автоматизацией"""
    
    def __init__(self, sample_picker):
        self.picker = sample_picker
        self.config = Config()
        
    def create_professional_mix(self, track_data, output_dir="output"):
        """Создание профессионального микса с эффектами"""
        os.makedirs(output_dir, exist_ok=True)
        
        tempo = track_data.get("tempo", 120)
        structure = track_data.get("structure", [])
        tracks = track_data.get("tracks", [])
        genre = track_data.get("genre", "electronic")
        
        # Подготовка timeline
        total_duration_sec = sum(s["duration"] for s in structure)
        total_duration_ms = int(total_duration_sec * 1000)
        
        # Создание мастер-шины
        master_mix = AudioSegment.silent(duration=total_duration_ms)
        
        # Создание групповых шин
        drum_bus = AudioSegment.silent(duration=total_duration_ms)
        bass_bus = AudioSegment.silent(duration=total_duration_ms)
        melody_bus = AudioSegment.silent(duration=total_duration_ms)
        fx_bus = AudioSegment.silent(duration=total_duration_ms)
        
        # Обработка каждого трека
        processed_tracks = []
        
        for track_idx, track in enumerate(tracks):
            track_result = self.process_single_track(
                track, tempo, total_duration_ms, genre, output_dir
            )
            
            if track_result:
                processed_tracks.append(track_result)
                
                # Роутинг в соответствующую шину
                track_audio = track_result["audio"]
                track_type = self.classify_track_type(track["sample_tags"])
                
                if track_type == "drums":
                    drum_bus = drum_bus.overlay(track_audio)
                elif track_type == "bass":
                    bass_bus = bass_bus.overlay(track_audio)
                elif track_type == "melody":
                    melody_bus = melody_bus.overlay(track_audio)
                else:
                    fx_bus = fx_bus.overlay(track_audio)
        
        # Обработка групповых шин
        drum_bus = self.process_drum_bus(drum_bus, genre)
        bass_bus = self.process_bass_bus(bass_bus, genre)
        melody_bus = self.process_melody_bus(melody_bus, genre)
        fx_bus = self.process_fx_bus(fx_bus, genre)
        
        # Сведение в мастер
        master_mix = master_mix.overlay(drum_bus)
        master_mix = master_mix.overlay(bass_bus)
        master_mix = master_mix.overlay(melody_bus)
        master_mix = master_mix.overlay(fx_bus)
        
        # Мастеринг
        master_mix = self.master_processing(master_mix, genre)
        
        # Сохранение результата
        final_path = os.path.join(output_dir, "final_professional_mix.wav")
        master_mix.export(final_path, format="wav")
        
        # Сохранение стемов
        self.export_stems(
            {"drums": drum_bus, "bass": bass_bus, "melody": melody_bus, "fx": fx_bus},
            output_dir
        )
        
        logging.info(f"🎛️ Профессиональный микс готов: {final_path}")
        return master_mix, final_path

    def process_single_track(self, track, tempo, total_duration_ms, genre, output_dir):
        """Обработка одного трека"""
        track_name = track.get("name", "unknown")
        sample_tags = track.get("sample_tags", [])
        volume = track.get("volume", -6)
        
        # Тайминг
        starts_at_beats = track.get("starts_at", 0)
        ends_at_beats = track.get("ends_at", None)
        
        beat_duration = 60.0 / tempo
        starts_at_ms = int(starts_at_beats * beat_duration * 1000)
        ends_at_ms = int(ends_at_beats * beat_duration * 1000) if ends_at_beats else total_duration_ms
        
        track_duration_ms = ends_at_ms - starts_at_ms
        
        # Подбор сэмплов
        picked_samples = self.picker.pick_samples_enhanced(
            required_tags=sample_tags,
            target_tempo=tempo,
            genre_hint=genre,
            top_k=3
        )
        
        if not picked_samples:
            logging.warning(f"⚠️ Не найдены сэмплы для {track_name}")
            return None
        
        # Выбор и обработка сэмпла
        chosen_sample = picked_samples[0]
        sample_path = chosen_sample["path"]
        
        try:
            sample_audio = AudioSegment.from_file(sample_path)
            
            # Интеллектуальная подгонка длительности
            processed_audio = self.smart_duration_fitting(
                sample_audio, track_duration_ms, chosen_sample.get("category", "oneshot")
            )
            
            # Применение volume
            processed_audio = processed_audio + volume
            
            # Позиционирование
            positioned_audio = AudioSegment.silent(duration=total_duration_ms)
            positioned_audio = positioned_audio.overlay(processed_audio, position=starts_at_ms)
            
            # Сохранение stem
            stem_path = os.path.join(output_dir, "stems", f"{track_name}.wav")
            os.makedirs(os.path.dirname(stem_path), exist_ok=True)
            positioned_audio.export(stem_path, format="wav")
            
            logging.info(f"✅ [{track_name}] обработан: {chosen_sample['filename']}")
            
            return {
                "name": track_name,
                "audio": positioned_audio,
                "sample_info": chosen_sample,
                "stem_path": stem_path
            }
            
        except Exception as e:
            logging.error(f"❌ Ошибка обработки {track_name}: {e}")
            return None

    def smart_duration_fitting(self, audio, target_duration_ms, category):
        """Интеллектуальная подгонка длительности"""
        current_duration = len(audio)
        
        if category == "loop":
            # Для лупов - зацикливание или обрезка
            if current_duration < target_duration_ms:
                repeats = (target_duration_ms // current_duration) + 1
                audio = (audio * repeats)[:target_duration_ms]
            else:
                audio = audio[:target_duration_ms]
        else:
            # Для one-shots - размещение + тишина или повторение
            if current_duration < target_duration_ms:
                # Если очень короткий - можем повторить несколько раз
                if current_duration < 2000:  # меньше 2 секунд
                    repeats = min(target_duration_ms // current_duration, 4)
                    audio = audio * repeats
                
                # Дополняем тишиной до нужной длины
                remaining = target_duration_ms - len(audio)
                if remaining > 0:
                    audio += AudioSegment.silent(duration=remaining)
            else:
                # Если слишком длинный - берём начало
                audio = audio[:target_duration_ms]
        
        return audio

    def classify_track_type(self, tags):
        """Классификация трека для роутинга в шины"""
        drums_tags = ["kick", "snare", "hihat", "clap", "percussion", "drums"]
        bass_tags = ["bass", "808", "sub", "reese"]
        melody_tags = ["lead", "melody", "synth", "piano", "guitar", "vocal"]
        fx_tags = ["fx", "effect", "sweep", "riser", "impact"]
        
        if any(tag in tags for tag in drums_tags):
            return "drums"
        elif any(tag in tags for tag in bass_tags):
            return "bass"
        elif any(tag in tags for tag in melody_tags):
            return "melody"
        else:
            return "fx"

    def process_drum_bus(self, drum_audio, genre):
        """Обработка барабанной шины"""
        if len(drum_audio.get_array_of_samples()) == 0:
            return drum_audio
            
        # Компрессия для punch
        drum_audio = self.apply_compression(drum_audio, ratio=4, threshold=-12)
        
        # EQ для ударных
        if genre in ["trap", "hip-hop"]:
            # Boost низы и верха
            drum_audio = drum_audio.low_pass_filter(12000).high_pass_filter(40)
        elif genre == "techno":
            # Более агрессивная обработка
            drum_audio = drum_audio.high_pass_filter(60)
        
        return drum_audio

    def process_bass_bus(self, bass_audio, genre):
        """Обработка басовой шины"""
        if len(bass_audio.get_array_of_samples()) == 0:
            return bass_audio
            
        # Фильтрация высоких частот
        bass_audio = bass_audio.low_pass_filter(250)
        
        # Компрессия для плотности
        bass_audio = self.apply_compression(bass_audio, ratio=6, threshold=-18)
        
        # Сатурация для жанров
        if genre in ["trap", "dubstep"]:
            bass_audio = self.apply_saturation(bass_audio, 0.3)
        
        return bass_audio

    def process_melody_bus(self, melody_audio, genre):
        """Обработка мелодической шины"""
        if len(melody_audio.get_array_of_samples()) == 0:
            return melody_audio
            
        # Мягкая компрессия
        melody_audio = self.apply_compression(melody_audio, ratio=2.5, threshold=-16)
        
        # Реверб для пространства
        melody_audio = self.apply_reverb(melody_audio, room_size=0.5, damping=0.3)
        
        # EQ в зависимости от жанра
        if genre == "ambient":
            melody_audio = melody_audio.high_pass_filter(200)
        elif genre in ["trap", "hip-hop"]:
            melody_audio = melody_audio.low_pass_filter(8000)
        
        return melody_audio

    def process_fx_bus(self, fx_audio, genre):
        """Обработка шины эффектов"""
        if len(fx_audio.get_array_of_samples()) == 0:
            return fx_audio
            
        # Стереорасширение
        fx_audio = self.apply_stereo_widening(fx_audio, 1.5)
        
        # Реверб для атмосферы
        fx_audio = self.apply_reverb(fx_audio, room_size=0.8, damping=0.5)
        
        return fx_audio

    def master_processing(self, audio, genre):
        """Мастеринг финального микса"""
        # Нормализация
        audio = effects.normalize(audio)
        
        # Мультибэнд компрессия (эмуляция)
        low_band = audio.low_pass_filter(250)
        mid_band = audio.high_pass_filter(250).low_pass_filter(4000)
        high_band = audio.high_pass_filter(4000)
        
        low_band = self.apply_compression(low_band, ratio=3, threshold=-15)
        mid_band = self.apply_compression(mid_band, ratio=2, threshold=-12)
        high_band = self.apply_compression(high_band, ratio=2.5, threshold=-10)
        
        # Сборка обратно
        audio = low_band.overlay(mid_band).overlay(high_band)
        
        # Лимитинг
        audio = self.apply_limiter(audio, threshold=-1)
        
        # Жанровая обработка
        if genre == "trap":
            audio = audio + 2  # Громче для trap
        elif genre == "ambient":
            audio = audio - 3  # Тише для ambient
        
        return audio

    def apply_compression(self, audio, ratio=4, threshold=-12, attack=10, release=100):
        """Эмуляция компрессии через динамическую обработку"""
        # Простая реализация компрессии
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        if len(samples) == 0:
            return audio
            
        # Нормализация
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # Детекция уровня
        if samples.ndim == 2:
            level = np.sqrt(np.mean(samples**2, axis=1))
        else:
            level = np.abs(samples)
        
        # Применение компрессии там, где уровень превышает threshold
        threshold_linear = 10**(threshold/20)
        
        compression_mask = level > threshold_linear
        if np.any(compression_mask):
            if samples.ndim == 2:
                reduction = ((level[compression_mask] / threshold_linear) ** (1/ratio - 1))
                samples[compression_mask] *= reduction[:, np.newaxis]
            else:
                reduction = ((level[compression_mask] / threshold_linear) ** (1/ratio - 1))
                samples[compression_mask] *= reduction
        
        # Конвертация обратно
        samples = np.clip(samples, -1, 1)
        samples = (samples * 32767).astype(np.int16)
        
        return audio._spawn(samples.tobytes())

    def apply_saturation(self, audio, amount=0.2):
        """Применение сатурации для теплоты"""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        # Мягкая сатурация
        saturated = np.tanh(samples * (1 + amount))
        saturated = np.clip(saturated * 32767, -32767, 32767).astype(np.int16)
        
        return audio._spawn(saturated.tobytes())

    def apply_reverb(self, audio, room_size=0.5, damping=0.3, wet_level=0.2):
        """Эмуляция реверба через delay и filtering"""
        # Простая эмуляция реверба
        delay_times = [50, 89, 134, 187, 267]  # мс
        
        reverb_audio = audio
        
        for delay_ms in delay_times:
            delay_samples = int((delay_ms / 1000) * audio.frame_rate)
            if delay_samples > 0:
                delayed = AudioSegment.silent(duration=delay_ms) + audio
                delayed = delayed.low_pass_filter(int(8000 * (1 - damping)))
                delayed = delayed - (20 + delay_ms/10)  # Затухание
                
                # Обрезаем до исходной длины
                delayed = delayed[:len(reverb_audio)]
                reverb_audio = reverb_audio.overlay(delayed)
        
        # Микс dry/wet
        dry_level = 1 - wet_level
        dry = audio.apply_gain(20 * math.log10(dry_level)) if dry_level > 0 else AudioSegment.silent(duration=len(audio))
        wet = reverb_audio.apply_gain(20 * math.log10(wet_level)) if wet_level > 0 else AudioSegment.silent(duration=len(reverb_audio))
        return dry.overlay(wet)

    def apply_stereo_widening(self, audio, width=1.5):
        """Расширение стереобазы"""
        if audio.channels != 2:
            return audio
            
        # Конвертация в numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples.reshape((-1, 2))
        
        # M/S обработка
        mid = (samples[:, 0] + samples[:, 1]) / 2
        side = (samples[:, 0] - samples[:, 1]) / 2
        
        # Расширение side
        side *= width
        
        # Обратная конвертация
        left = mid + side
        right = mid - side
        
        # Нормализация и клиппинг
        stereo = np.column_stack([left, right])
        stereo = np.clip(stereo, -1, 1)
        stereo = (stereo * 32767).astype(np.int16)
        
        return audio._spawn(stereo.tobytes())

    def apply_limiter(self, audio, threshold=-1, release=50):
        """Лимитер для предотвращения клиппинга"""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        threshold_linear = 10**(threshold/20)
        
        # Простой hard limiter
        samples = np.clip(samples, -threshold_linear, threshold_linear)
        
        # Конвертация обратно
        samples = (samples * 32767).astype(np.int16)
        return audio._spawn(samples.tobytes())

    def export_stems(self, stem_dict, output_dir):
        """Экспорт стемов (групповых шин)"""
        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)
        
        for stem_name, stem_audio in stem_dict.items():
            if len(stem_audio.get_array_of_samples()) > 0:
                stem_path = os.path.join(stems_dir, f"{stem_name}_bus.wav")
                stem_audio.export(stem_path, format="wav")
                logging.info(f"💾 Стем сохранён: {stem_path}")

class AutomationEngine:
    """Движок автоматизации параметров"""
    
    @staticmethod
    def create_volume_automation(audio, automation_points):
        """Создание автоматизации громкости"""
        # automation_points: [(time_ms, volume_db), ...]
        
        if len(automation_points) < 2:
            return audio
        
        segments = []
        duration = len(audio)
        
        for i in range(len(automation_points) - 1):
            start_time, start_vol = automation_points[i]
            end_time, end_vol = automation_points[i + 1]
            
            if start_time >= duration:
                break
                
            # Извлекаем сегмент
            segment_start = max(0, start_time)
            segment_end = min(duration, end_time)
            segment = audio[segment_start:segment_end]
            
            # Применяем градуальное изменение громкости
            if len(segment) > 0:
                segment = AutomationEngine.apply_volume_curve(segment, start_vol, end_vol)
                segments.append(segment)
        
        # Собираем обратно
        if segments:
            return sum(segments)
        return audio

    @staticmethod
    def apply_volume_curve(audio, start_db, end_db):
        """Применение плавного изменения громкости"""
        length = len(audio)
        if length == 0:
            return audio
            
        # Создаём кривую изменения
        volume_curve = np.linspace(start_db, end_db, length)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Применяем кривую
        for i, vol_db in enumerate(volume_curve):
            multiplier = 10**(vol_db/20)
            if audio.channels == 2:
                if i*2 + 1 < len(samples):
                    samples[i*2] *= multiplier
                    samples[i*2 + 1] *= multiplier
            else:
                if i < len(samples):
                    samples[i] *= multiplier
        
        samples = np.clip(samples, -32767, 32767).astype(np.int16)
        return audio._spawn(samples.tobytes())

class ArrangementEngine:
    """Движок для создания профессиональных аранжировок"""
    
    def __init__(self, config):
        self.config = config
    
    def create_dynamic_arrangement(self, base_tracks, genre, mood, total_duration):
        """Создание динамической аранжировки"""
        arranged_tracks = []
        beat_duration = 60.0 / base_tracks[0].get("tempo", 120)
        total_beats = int(total_duration / beat_duration)
        
        # Создание секций с различной интенсивностью
        sections = self.generate_intensity_sections(genre, total_beats)
        
        for track in base_tracks:
            arranged_track = track.copy()
            
            # Определение паттерна появления трека
            track_pattern = self.get_track_pattern(track["sample_tags"], genre, sections)
            
            arranged_track.update(track_pattern)
            arranged_tracks.append(arranged_track)
        
        return arranged_tracks

    def generate_intensity_sections(self, genre, total_beats):
        """Генерация секций с разной интенсивностью"""
        if genre == "trap":
            return [
                {"type": "intro", "beats": (0, 32), "intensity": 0.3},
                {"type": "verse", "beats": (32, 64), "intensity": 0.6},
                {"type": "hook", "beats": (64, 96), "intensity": 1.0},
                {"type": "verse2", "beats": (96, 128), "intensity": 0.7},
                {"type": "hook2", "beats": (128, 160), "intensity": 1.0},
                {"type": "outro", "beats": (160, total_beats), "intensity": 0.4}
            ]
        elif genre == "house":
            return [
                {"type": "intro", "beats": (0, 64), "intensity": 0.4},
                {"type": "build", "beats": (64, 96), "intensity": 0.7},
                {"type": "drop", "beats": (96, 224), "intensity": 1.0},
                {"type": "breakdown", "beats": (224, 288), "intensity": 0.5},
                {"type": "drop2", "beats": (288, 416), "intensity": 1.0},
                {"type": "outro", "beats": (416, total_beats), "intensity": 0.3}
            ]
        else:
            # Generic electronic
            section_length = total_beats // 4
            return [
                {"type": "intro", "beats": (0, section_length), "intensity": 0.4},
                {"type": "build", "beats": (section_length, section_length*2), "intensity": 0.7},
                {"type": "climax", "beats": (section_length*2, section_length*3), "intensity": 1.0},
                {"type": "outro", "beats": (section_length*3, total_beats), "intensity": 0.4}
            ]

    def get_track_pattern(self, tags, genre, sections):
        """Определение паттерна появления трека"""
        track_type = self.classify_track_role(tags)
        
        patterns = {
            "drums": {
                "intro": 0.5, "verse": 1.0, "hook": 1.0, "build": 1.0, "drop": 1.0
            },
            "bass": {
                "intro": 0.0, "verse": 0.8, "hook": 1.0, "build": 0.6, "drop": 1.0
            },
            "melody": {
                "intro": 0.3, "verse": 0.6, "hook": 1.0, "build": 0.8, "drop": 1.0
            },
            "fx": {
                "intro": 0.2, "verse": 0.3, "hook": 0.5, "build": 1.0, "drop": 0.7
            }
        }
        
        base_pattern = patterns.get(track_type, patterns["melody"])
        
        # Создание тайминга на основе секций
        timing = {"starts_at": 0, "automation": []}
        
        for section in sections:
            section_start_beat = section["beats"][0]
            section_intensity = section["intensity"]
            track_presence = base_pattern.get(section["type"], 0.5)
            
            final_volume = -20 + (track_presence * section_intensity * 20)
            timing["automation"].append((section_start_beat, final_volume))
        
        return timing

    def classify_track_role(self, tags):
        """Классификация роли трека в миксе"""
        if any(tag in ["kick", "snare", "hihat", "drums"] for tag in tags):
            return "drums"
        elif any(tag in ["bass", "808", "sub"] for tag in tags):
            return "bass"
        elif any(tag in ["lead", "melody", "synth", "vocal"] for tag in tags):
            return "melody"
        else:
            return "fx"

# Утилиты для экспорта в различные форматы
class ExportManager:
    """Менеджер экспорта в различные форматы"""
    
    @staticmethod
    def export_to_wav(audio, path, quality="high"):
        """Экспорт в WAV с различным качеством"""
        if quality == "high":
            audio.export(path, format="wav", parameters=["-ar", "48000", "-ac", "2"])
        elif quality == "medium":
            audio.export(path, format="wav", parameters=["-ar", "44100", "-ac", "2"])
        else:
            audio.export(path, format="wav")
    
    @staticmethod
    def export_to_mp3(audio, path, bitrate="320k"):
        """Экспорт в MP3"""
        audio.export(path, format="mp3", bitrate=bitrate)
    
    @staticmethod
    def export_stems_package(stems_dict, output_dir):
        """Экспорт пакета стемов"""
        stems_dir = os.path.join(output_dir, "stems_package")
        os.makedirs(stems_dir, exist_ok=True)
        
        for stem_name, stem_audio in stems_dict.items():
            # WAV для профессионального использования
            wav_path = os.path.join(stems_dir, f"{stem_name}.wav")
            ExportManager.export_to_wav(stem_audio, wav_path, "high")
            
            # MP3 для демо
            mp3_path = os.path.join(stems_dir, f"{stem_name}_demo.mp3")
            ExportManager.export_to_mp3(stem_audio, mp3_path, "192k")

# Интеграция с внешними сервисами
class ExternalIntegration:
    """Интеграция с внешними сервисами и API"""
    
    @staticmethod
    def upload_to_soundcloud(file_path, title, description="", tags=None):
        """Заглушка для загрузки в SoundCloud"""
        # Здесь будет интеграция с SoundCloud API
        logging.info(f"🎵 Загрузка в SoundCloud: {title}")
        return f"https://soundcloud.com/user/track-{hash(title)}"
    
    @staticmethod  
    def analyze_with_ai(audio_path):
        """Заглушка для AI-анализа трека"""
        # Здесь может быть интеграция с сервисами анализа музыки
        return {
            "genre_confidence": 0.85,
            "energy_level": 0.7,
            "danceability": 0.8,
            "valence": 0.6
        }

# Система кэширования для оптимизации
class CacheManager:
    """Менеджер кэша для ускорения повторных операций"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_audio_cache_key(self, file_path, operation="analysis"):
        """Генерация ключа кэша для аудиофайла"""
        import hashlib
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_size}_{stat.st_mtime}_{operation}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def cache_analysis_result(self, file_path, result):
        """Кэширование результата анализа"""
        cache_key = self.get_audio_cache_key(file_path)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logging.warning(f"Ошибка сохранения кэша: {e}")
    
    def get_cached_analysis(self, file_path):
        """Получение закэшированного результата"""
        cache_key = self.get_audio_cache_key(file_path)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Ошибка загрузки кэша: {e}")
        
        return None

# Система плагинов для расширения функциональности
class PluginManager:
    """Менеджер плагинов для расширения возможностей"""
    
    def __init__(self):
        self.plugins = {}
        self.load_builtin_plugins()
    
    def load_builtin_plugins(self):
        """Загрузка встроенных плагинов"""
        self.register_plugin("tempo_detection", self.advanced_tempo_detection)
        self.register_plugin("genre_classification", self.ai_genre_classifier)
        self.register_plugin("mood_analysis", self.mood_analyzer)
    
    def register_plugin(self, name, function):
        """Регистрация плагина"""
        self.plugins[name] = function
        logging.info(f"🔌 Плагин зарегистрирован: {name}")
    
    def run_plugin(self, name, *args, **kwargs):
        """Запуск плагина"""
        if name in self.plugins:
            return self.plugins[name](*args, **kwargs)
        else:
            logging.warning(f"⚠️ Плагин не найден: {name}")
            return None
    
    def advanced_tempo_detection(self, audio_path):
        """Продвинутое определение темпа"""
        try:
            y, sr = librosa.load(audio_path, duration=30)
            
            # Несколько алгоритмов для точности
            tempo1, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Альтернативный метод через onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                intervals = np.diff(onset_times)
                if len(intervals) > 0:
                    avg_interval = np.median(intervals)
                    tempo2 = 60.0 / avg_interval
                else:
                    tempo2 = tempo1
            else:
                tempo2 = tempo1
            
            # Объединение результатов
            final_tempo = np.mean([tempo1, tempo2])
            
            # Корректировка для типичных диапазонов
            if final_tempo < 70:
                final_tempo *= 2
            elif final_tempo > 200:
                final_tempo /= 2
                
            return {"tempo": float(final_tempo), "confidence": 0.8}
            
        except Exception as e:
            logging.warning(f"Ошибка определения темпа: {e}")
            return {"tempo": 120, "confidence": 0.1}
    
    def ai_genre_classifier(self, audio_path):
        """AI классификация жанра (заглушка)"""
        # Здесь может быть интеграция с ML моделью
        return {
            "genre": "electronic",
            "confidence": 0.5,
            "sub_genres": ["ambient", "techno", "house"]
        }
    
    def mood_analyzer(self, audio_path):
        """Анализ настроения трека"""
        try:
            y, sr = librosa.load(audio_path, duration=30)
            
            # Анализ спектральных характеристик
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            energy = librosa.feature.rms(y=y).mean()
            
            # Эвристическое определение настроения
            mood_scores = {}
            
            # Энергичность
            if energy > 0.1 and tempo > 120:
                mood_scores["energetic"] = 0.8
            
            # Темнота (низкие частоты + медленный темп)
            if spectral_centroid < 2000 and tempo < 100:
                mood_scores["dark"] = 0.7
            
            # Мелодичность (средние частоты + умеренная энергия)
            if 2000 < spectral_centroid < 6000 and 0.05 < energy < 0.15:
                mood_scores["melodic"] = 0.6
            
            # Агрессивность (высокая энергия + высокие частоты)
            if energy > 0.15 and spectral_centroid > 4000:
                mood_scores["aggressive"] = 0.8
            
            return mood_scores
            
        except Exception as e:
            logging.warning(f"Ошибка анализа настроения: {e}")
            return {"neutral": 0.5}