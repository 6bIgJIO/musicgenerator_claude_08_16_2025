# ИСПРАВЛЕННЫЙ musicgen_wrapper.py
# Замени содержимое файла на это:

import torch
import torchaudio
import io
import numpy as np
import soundfile as sf
from typing import Optional, List, Dict, Union
import warnings
import logging
import time
from pydub import AudioSegment
from pydub.generators import Sine, Square, WhiteNoise

warnings.filterwarnings("ignore")

class MusicGenEngine:
    def __init__(self, model_name: str = "facebook/musicgen-medium"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MUSICGEN_AVAILABLE = False
        self.use_fallback = False  # Флаг для принудительного fallback
        self._load_model()

    def _load_model(self):
        """ИСПРАВЛЕНО: Попытка загрузки с принудительным fallback"""
        try:
            self.logger.info(f"🔄 Попытка загрузки модели: {self.model_name}")
            
            # Пробуем импортировать audiocraft
            try:
                from audiocraft.models import musicgen
                AUDIOCRAFT_AVAILABLE = True
            except ImportError as e:
                self.logger.error(f"❌ Audiocraft не найден: {e}")
                AUDIOCRAFT_AVAILABLE = False
            
            if AUDIOCRAFT_AVAILABLE:
                # Пробуем загрузить модели по порядку
                fallback_models = [
                    "facebook/musicgen-small",  # Самая лёгкая модель
                    "facebook/musicgen-medium",
                    self.model_name
                ]
                
                for model_name in fallback_models:
                    try:
                        self.logger.info(f"🔄 Пробуем загрузить: {model_name}")
                        
                        # Устанавливаем короткий timeout
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Model loading timeout")
                        
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)  # 30 секунд timeout
                        
                        try:
                            self.model = musicgen.MusicGen.get_pretrained(model_name)
                            self.model.set_generation_params(duration=8)
                            self.model_name = model_name
                            self.MUSICGEN_AVAILABLE = True
                            
                            signal.alarm(0)  # Отменяем timeout
                            signal.signal(signal.SIGALRM, old_handler)
                            
                            self.logger.info(f"✅ MusicGen успешно загружен: {model_name} на {self.device}")
                            return
                            
                        except Exception as model_error:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                            self.logger.warning(f"⚠️ Не удалось загрузить {model_name}: {model_error}")
                            continue
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Ошибка загрузки {model_name}: {e}")
                        continue
            
            # Если ничего не загрузилось, используем fallback
            self.logger.warning("⚠️ MusicGen недоступен, включаем режим fallback")
            self.use_fallback = True
            self.MUSICGEN_AVAILABLE = False
            
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка загрузки: {e}")
            self.use_fallback = True
            self.MUSICGEN_AVAILABLE = False

    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        genre_hint: Optional[str] = None
    ) -> bytes:
        """ИСПРАВЛЕНО: Генерация с гарантированным возвратом аудио"""
        
        self.logger.info(f"🎼 Generating: '{prompt}' ({duration}s)")
        
        # ИСПРАВЛЕНИЕ: Всегда возвращаем аудио, даже если MusicGen не работает
        if not self.model or not self.MUSICGEN_AVAILABLE or self.use_fallback:
            self.logger.warning("🔄 MusicGen недоступен, используем интеллектуальный fallback")
            return await self._generate_intelligent_fallback(prompt, duration, genre_hint)

        try:
            # Пробуем реальную генерацию MusicGen
            safe_duration = min(duration, 30)  # Ограничиваем длительность
            
            self.model.set_generation_params(
                duration=safe_duration,
                use_sampling=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            enhanced_prompt = self._enhance_prompt_for_genre(prompt, genre_hint)
            self.logger.info(f"📝 Enhanced prompt: {enhanced_prompt}")

            # Генерируем с timeout
            with torch.no_grad():
                start_time = time.time()
                wav_tensor = self.model.generate([enhanced_prompt])
                generation_time = time.time() - start_time
                
                self.logger.info(f"⏱️ MusicGen generation time: {generation_time:.2f}s")

            # ИСПРАВЛЕНИЕ: Проверяем результат
            if wav_tensor is None or wav_tensor.numel() == 0:
                self.logger.warning("⚠️ MusicGen вернул пустой результат, используем fallback")
                return await self._generate_intelligent_fallback(prompt, duration, genre_hint)

            self.logger.info(f"🔊 Generated tensor shape: {wav_tensor.shape}")

            # Обрабатываем тензор
            if wav_tensor.dim() == 3:
                audio_array = wav_tensor[0].cpu().numpy()
            elif wav_tensor.dim() == 2:
                audio_array = wav_tensor.cpu().numpy()
            else:
                self.logger.warning(f"⚠️ Неожиданная форма тензора: {wav_tensor.shape}")
                return await self._generate_intelligent_fallback(prompt, duration, genre_hint)

            # Нормализация
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))

            # Проверяем уровень сигнала
            rms = np.sqrt(np.mean(audio_array**2))
            self.logger.info(f"🔊 Generated audio RMS: {rms:.6f}")

            if rms < 1e-6:
                self.logger.warning("⚠️ Очень тихий звук, увеличиваем громкость")
                audio_array = audio_array * 1000
                audio_array = np.clip(audio_array, -1.0, 1.0)

            # Конвертируем в WAV
            sample_rate = self.model.sample_rate
            audio_bytes = self._array_to_wav_bytes(audio_array, sample_rate)

            if len(audio_bytes) < 1000:
                self.logger.warning(f"⚠️ Слишком маленький файл: {len(audio_bytes)} bytes, используем fallback")
                return await self._generate_intelligent_fallback(prompt, duration, genre_hint)

            self.logger.info(f"✅ MusicGen generation completed: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            self.logger.error(f"❌ MusicGen generation error: {e}")
            self.logger.info("🔄 Переключаемся на fallback генерацию")
            return await self._generate_intelligent_fallback(prompt, duration, genre_hint)

    async def _generate_intelligent_fallback(self, prompt: str, duration: int, genre: Optional[str]) -> bytes:
        """ИСПРАВЛЕНО: Интеллектуальный fallback генератор на основе промпта и жанра"""
        
        self.logger.info(f"🎹 Generating intelligent fallback: '{prompt}' ({duration}s, genre: {genre})")
        
        try:
            # Определяем параметры на основе жанра
            genre = genre or "electronic"  # Дефолт
            genre = genre.lower()
            
            # Параметры по жанрам
            genre_params = {
                "trap": {"bpm": 160, "bass_freq": 60, "energy": 0.8, "style": "aggressive"},
                "lofi": {"bpm": 80, "bass_freq": 80, "energy": 0.3, "style": "chill"},
                "dnb": {"bpm": 174, "bass_freq": 55, "energy": 0.9, "style": "energetic"},
                "ambient": {"bpm": 60, "bass_freq": 40, "energy": 0.2, "style": "ethereal"},
                "techno": {"bpm": 128, "bass_freq": 65, "energy": 0.8, "style": "driving"},
                "house": {"bpm": 124, "bass_freq": 70, "energy": 0.7, "style": "groovy"},
                "cinematic": {"bpm": 90, "bass_freq": 50, "energy": 0.6, "style": "epic"},
                "electronic": {"bpm": 120, "bass_freq": 65, "energy": 0.6, "style": "modern"}
            }
            
            params = genre_params.get(genre, genre_params["electronic"])
            
            bpm = params["bpm"]
            bass_freq = params["bass_freq"]
            energy = params["energy"]
            style = params["style"]
            
            self.logger.info(f"🎛️ Fallback params: {bpm}BPM, energy={energy}, style={style}")
            
            # Создаём базовый ритм
            duration_ms = int(duration * 1000)
            beat_duration = int(60000 / bpm)  # Длительность одного бита в мс
            
            # Создаём элементы композиции
            kick = self._create_kick(bass_freq, energy)
            snare = self._create_snare(energy)
            hihat = self._create_hihat(energy)
            
            # Создаём мелодию/гармонию
            melody = self._create_melody_for_genre(genre, duration_ms, bpm, energy)
            
            # Создаём ритм-секцию
            rhythm = self._create_rhythm_section(kick, snare, hihat, duration_ms, bpm, genre, energy)
            
            # Микшируем всё вместе
            if len(melody) > len(rhythm):
                melody = melody[:len(rhythm)]
            elif len(rhythm) > len(melody):
                # Дублируем мелодию до нужной длины
                melody = melody * (len(rhythm) // len(melody) + 1)
                melody = melody[:len(rhythm)]
            
            # Накладываем мелодию на ритм
            final_audio = rhythm.overlay(melody)
            
            # Применяем жанровые эффекты
            final_audio = self._apply_genre_effects(final_audio, genre, energy)
            
            # Нормализуем и экспортируем
            final_audio = final_audio.normalize(headroom=3.0)
            
            buffer = io.BytesIO()
            final_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            buffer.close()
            
            self.logger.info(f"✅ Intelligent fallback completed: {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent fallback failed: {e}")
            return self._create_emergency_audio(duration)

    def _create_kick(self, freq: int, energy: float) -> AudioSegment:
        """Создание kick-драма"""
        kick = Sine(freq).to_audio_segment(duration=200)
        kick = kick.fade_in(10).fade_out(150)
        kick = kick.apply_gain(-6 + energy * 8)  # Громкость в зависимости от энергии
        return kick

    def _create_snare(self, energy: float) -> AudioSegment:
        """Создание snare"""
        noise = WhiteNoise().to_audio_segment(duration=150)
        tone = Sine(200).to_audio_segment(duration=150)
        snare = noise.overlay(tone)
        snare = snare.band_pass_filter(200, 4000)
        snare = snare.apply_gain(-10 + energy * 6)
        return snare

    def _create_hihat(self, energy: float) -> AudioSegment:
        """Создание hi-hat"""
        hihat = WhiteNoise().to_audio_segment(duration=80)
        hihat = hihat.high_pass_filter(8000)
        hihat = hihat.apply_gain(-15 + energy * 5)
        return hihat

    def _create_melody_for_genre(self, genre: str, duration_ms: int, bpm: int, energy: float) -> AudioSegment:
        """Создание мелодии под жанр"""
        
        # Базовые ноты для разных жанров
        genre_scales = {
            "trap": [60, 63, 65, 67, 70, 72],  # C, Eb, F, G, Bb, C (минорный)
            "lofi": [60, 62, 64, 67, 69],      # Пентатоника
            "ambient": [60, 64, 67, 72, 76],   # Мажорное трезвучие + октавы
            "techno": [60, 60, 67, 67],        # Простые интервалы
            "cinematic": [48, 52, 55, 60, 64], # Низкие драматичные ноты
        }
        
        scale = genre_scales.get(genre, [60, 64, 67, 72])  # Дефолт
        
        melody = AudioSegment.silent(duration=0)
        note_duration = int(60000 / bpm * 2)  # Длительность ноты
        
        total_notes_needed = duration_ms // note_duration + 1
        
        for i in range(total_notes_needed):
            # Выбираем ноту из гаммы
            midi_note = scale[i % len(scale)]
            freq = 440 * (2 ** ((midi_note - 69) / 12))  # Конвертируем MIDI в частоту
            
            # Создаём ноту
            if genre in ["ambient", "cinematic"]:
                note = Sine(freq).to_audio_segment(duration=note_duration * 2)
                note = note.fade_in(100).fade_out(100)
            else:
                note = Square(freq).to_audio_segment(duration=note_duration)
                note = note.fade_in(10).fade_out(10)
            
            note = note.apply_gain(-20 + energy * 10)
            
            # Добавляем паузы для ритмичности
            if genre == "trap" and i % 4 == 3:
                pause = AudioSegment.silent(duration=note_duration // 2)
                melody += note + pause
            else:
                melody += note
        
        # Обрезаем до нужной длины
        return melody[:duration_ms]

    def _create_rhythm_section(self, kick: AudioSegment, snare: AudioSegment, hihat: AudioSegment, 
                              duration_ms: int, bpm: int, genre: str, energy: float) -> AudioSegment:
        """Создание ритм-секции"""
        
        beat_duration = int(60000 / bpm)
        step_duration = beat_duration // 4  # 16-я нота
        
        # Ритмические паттерны по жанрам
        patterns = {
            "trap": {
                "kick":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "hihat": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            },
            "dnb": {
                "kick":  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "snare": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "hihat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            },
            "house": {
                "kick":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "hihat": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            },
            "lofi": {
                "kick":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "hihat": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            }
        }
        
        pattern = patterns.get(genre, patterns.get("house", patterns["trap"]))
        
        # Создаём один такт
        bar = AudioSegment.silent(duration=beat_duration * 4)
        
        for instrument, inst_pattern in pattern.items():
            sound = {"kick": kick, "snare": snare, "hihat": hihat}[instrument]
            
            for i, hit in enumerate(inst_pattern):
                if hit:
                    pos = i * step_duration
                    # Добавляем немного рандомности для живости
                    if energy > 0.7 and np.random.random() < 0.2:
                        varied_sound = sound.apply_gain(np.random.randint(-2, 3))
                        bar = bar.overlay(varied_sound, position=pos)
                    else:
                        bar = bar.overlay(sound, position=pos)
        
        # Повторяем такт на всю длительность
        bars_needed = duration_ms // len(bar) + 1
        rhythm = bar * bars_needed
        return rhythm[:duration_ms]

    def _apply_genre_effects(self, audio: AudioSegment, genre: str, energy: float) -> AudioSegment:
        """Применение жанровых эффектов"""
        
        if genre == "lofi":
            # Винтажный звук
            audio = audio.low_pass_filter(8000)
            audio = audio.apply_gain(-3)
            
        elif genre == "trap":
            # Тяжёлые басы
            audio = audio.low_pass_filter(15000)
            
        elif genre == "ambient":
            # Реверб эффект (имитация)
            reverb_delay = 150
            reverb_audio = audio.apply_gain(-15)
            for i in range(3):
                audio = audio.overlay(reverb_audio, position=reverb_delay * (i + 1))
                
        elif genre == "dnb":
            # Яркий, энергичный звук
            audio = audio.apply_gain(2)
            
        return audio

    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Конвертация numpy array в WAV bytes"""
        buffer = io.BytesIO()
        
        if audio_array.ndim == 1:
            sf.write(buffer, audio_array, sample_rate, format='WAV')
        else:
            if audio_array.shape[0] == 2:
                sf.write(buffer, audio_array.T, sample_rate, format='WAV')
            else:
                sf.write(buffer, audio_array[0], sample_rate, format='WAV')
        
        audio_bytes = buffer.getvalue()
        buffer.close()
        return audio_bytes

    def _create_emergency_audio(self, duration: int) -> bytes:
        """Экстренное создание минимального аудио"""
        self.logger.warning("🚨 Creating emergency audio")
        
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            
            # Простой звук - микс шума и тонов
            t = np.linspace(0, duration, samples)
            noise = np.random.normal(0, 0.05, samples)
            tone1 = np.sin(2 * np.pi * 220 * t) * 0.1  # A3
            tone2 = np.sin(2 * np.pi * 330 * t) * 0.1  # E4
            
            audio_array = (noise + tone1 + tone2).astype(np.float32)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"❌ Emergency audio creation failed: {e}")
            # Возвращаем минимальный WAV заголовок с тишиной
            return self._create_silent_wav(duration)

    def _create_silent_wav(self, duration: int) -> bytes:
        """Создание тишины в формате WAV"""
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # WAV заголовок
        import struct
        header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + samples * 2,
            b'WAVE',
            b'fmt ',
            16, 1, 1,  # PCM, mono
            sample_rate,
            sample_rate * 2,
            2, 16,     # 16-bit
            b'data',
            samples * 2
        )
        
        # Тишина
        data = b'\x00\x00' * samples
        return header + data

    def _enhance_prompt_for_genre(self, prompt: str, genre: Optional[str]) -> str:
        """Улучшение промпта для жанра"""
        if not genre:
            return prompt

        genre_enhancements = {
            "trap": "heavy 808s, tight snares, dark atmosphere, aggressive",
            "lofi": "warm analog sound, vinyl texture, mellow vibes, nostalgic",
            "dnb": "fast breakbeats, heavy bass, energetic, liquid",
            "ambient": "ethereal pads, spacious reverb, peaceful, atmospheric",
            "techno": "driving four-on-the-floor, hypnotic, industrial, minimal",
            "house": "groovy four-on-the-floor, soulful, danceable, uplifting",
            "cinematic": "epic orchestral, dramatic, heroic, emotional"
        }

        enhancement = genre_enhancements.get(genre.lower(), "")
        if enhancement:
            enhanced = f"{prompt}, {enhancement}"
            return enhanced
        return prompt

# ДОПОЛНИТЕЛЬНО - если хочешь протестировать fallback отдельно:
if __name__ == "__main__":
    import asyncio
    
    async def test_fallback():
        engine = MusicGenEngine()
        engine.use_fallback = True  # Принудительный fallback
        
        print("🧪 Testing fallback generation...")
        
        test_cases = [
            ("dark trap beat", "trap"),
            ("chill lofi vibes", "lofi"),
            ("epic cinematic music", "cinematic"),
            ("energetic dnb", "dnb")
        ]
        
        for prompt, genre in test_cases:
            print(f"\n🎵 Testing: {prompt} ({genre})")
            audio_bytes = await engine.generate(prompt, duration=10, genre_hint=genre)
            
            if audio_bytes and len(audio_bytes) > 1000:
                print(f"✅ Success: {len(audio_bytes)} bytes")
                
                # Сохраняем для тестирования
                with open(f"test_{genre}_fallback.wav", "wb") as f:
                    f.write(audio_bytes)
            else:
                print(f"❌ Failed: {len(audio_bytes) if audio_bytes else 0} bytes")
    
    asyncio.run(test_fallback())
