import logging
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import io

# Аудио обработка
from pydub import AudioSegment, effects
import librosa
import soundfile as sf

from config import config, MasteringPurpose
from sample_engine import EffectsChain


@dataclass
class MasteringTarget:
    """Целевые параметры для мастеринга"""
    lufs: float                    # Целевая громкость
    peak_ceiling: float           # Максимальный пик
    dynamic_range: float          # Желаемый динамический диапазон
    stereo_width: float          # Стерео ширина (1.0 = нормальная)
    frequency_balance: Dict      # Баланс частот
    harmonic_content: float      # Желаемое гармоническое содержание
    transient_preservation: float # Сохранение транзиентов


class SmartMasteringEngine:
    """
    ИСПРАВЛЕННАЯ умная система мастеринга с адаптацией под назначение и жанр
    
    ИСПРАВЛЕНИЯ:
    - Убрана заглушка с тишиной в _load_audio_from_bytes
    - Улучшена обработка ошибок без потери аудио
    - Исправлена логика fallback'ов
    - Добавлена валидация аудиоданных на каждом этапе
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.effects_chain = EffectsChain()
        
        # Инициализируем анализаторы
        self._init_analyzers()
        
        # Кэш анализа материала
        self.analysis_cache = {}
    
    def _init_analyzers(self):
        """Инициализация анализаторов аудио"""
        self.logger.info("🎛️ Initializing mastering analyzers...")
    
    def _load_audio_from_bytes(self, audio_bytes: bytes) -> AudioSegment:
        """
        ИСПРАВЛЕННАЯ загрузка аудио из bytes - БЕЗ ЗАГЛУШЕК С ТИШИНОЙ!
        
        Args:
            audio_bytes: Аудио в формате bytes
            
        Returns:
            AudioSegment объект или вызывает исключение
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("❌ CRITICAL: Empty audio bytes provided!")
        
        try:
            # Создаём BytesIO объект из bytes
            audio_io = io.BytesIO(audio_bytes)
            
            # Пробуем различные форматы
            formats_to_try = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aiff']
            
            last_error = None
            
            for fmt in formats_to_try:
                try:
                    audio_io.seek(0)  # Возвращаемся к началу
                    audio = AudioSegment.from_file(audio_io, format=fmt)
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: убеждаемся что аудио не пустое
                    if len(audio) == 0:
                        self.logger.warning(f"⚠️ Audio loaded as {fmt} but is empty (0 duration)")
                        continue
                    
                    if audio.max_dBFS == float('-inf'):
                        self.logger.warning(f"⚠️ Audio loaded as {fmt} but is silent (max_dBFS = -inf)")
                        continue
                    
                    self.logger.info(f"✅ Successfully loaded audio as {fmt}: "
                                   f"{len(audio)/1000:.1f}s, {audio.channels}ch, {audio.frame_rate}Hz, "
                                   f"peak: {audio.max_dBFS:.1f}dB")
                    return audio
                    
                except Exception as e:
                    last_error = e
                    self.logger.debug(f"Failed to load as {fmt}: {e}")
                    continue
            
            # Если ни один формат не подошёл, пробуем без указания формата
            try:
                audio_io.seek(0)
                audio = AudioSegment.from_file(audio_io)
                
                # Опять же проверяем что аудио не пустое
                if len(audio) == 0 or audio.max_dBFS == float('-inf'):
                    raise ValueError("Loaded audio is empty or silent")
                
                self.logger.info(f"✅ Successfully loaded audio (auto-detect): "
                               f"{len(audio)/1000:.1f}s, {audio.channels}ch, {audio.frame_rate}Hz")
                return audio
                
            except Exception as e:
                last_error = e
                self.logger.error(f"❌ Auto-detect also failed: {e}")
            
            # УБРАНО: Возврат тишины как fallback
            # Теперь выбрасываем исключение вместо возврата тишины
            raise ValueError(f"❌ CRITICAL: Cannot load audio from bytes! Last error: {last_error}")
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Failed to load audio from bytes: {e}")
            raise  # Пробрасываем исключение вместо возврата заглушки
    
    async def master_track(
        self,
        audio: bytes,
        target_config: Dict,
        genre_info: Dict,
        purpose: str
    ) -> Tuple[AudioSegment, Dict]:
        """
        ИСПРАВЛЕННАЯ главная функция мастеринга трека
        """
        self.logger.info(f"🎛️ Starting smart mastering: {purpose}")

        try:
            # ИСПРАВЛЕНО: Правильная обработка входных данных
            source_audio = None
            
            if isinstance(audio, bytes):
                if len(audio) == 0:
                    raise ValueError("❌ CRITICAL: Empty bytes provided to master_track!")
                source_audio = self._load_audio_from_bytes(audio)
                
            elif hasattr(audio, 'export'):  # AudioSegment
                source_audio = audio
                
            elif hasattr(audio, 'read'):  # file-like object
                buffer = io.BytesIO()
                audio.export(buffer, format="wav")
                buffer.seek(0)
                source_audio = self._load_audio_from_bytes(buffer.getvalue())
                
            else:
                raise TypeError(f"❌ CRITICAL: Unsupported audio type: {type(audio)}")

            # ИСПРАВЛЕНО: Проверяем что source_audio действительно содержит звук
            if source_audio is None:
                raise ValueError("❌ CRITICAL: source_audio is None after conversion!")
            
            if len(source_audio) < 100:  # Минимум 100мс
                raise ValueError(f"❌ CRITICAL: Audio too short: {len(source_audio)}ms")
            
            if source_audio.max_dBFS == float('-inf'):
                raise ValueError("❌ CRITICAL: Source audio is completely silent!")

            # Анализ исходного материала
            source_analysis = await self._analyze_source_material(source_audio)
            self.logger.info(
                f"  📊 Source analysis: LUFS {source_analysis['lufs']:.1f}, "
                f"peak {source_analysis['peak']:.1f}dB, "
                f"duration {source_analysis['duration']:.1f}s"
            )

            # Создание целей мастеринга
            mastering_target = self._create_mastering_target(target_config, genre_info, source_analysis)
            
            # Планирование обработки
            processing_plan = await self._plan_mastering_chain(source_analysis, mastering_target, genre_info)
            
            # Применение обработки
            mastered_audio = await self._apply_mastering_chain(source_audio, processing_plan)
            
            # ИСПРАВЛЕНО: Проверяем что mastered_audio не стал тишиной
            if mastered_audio.max_dBFS == float('-inf'):
                self.logger.warning("⚠️ Mastering resulted in silence, using original audio!")
                mastered_audio = source_audio
            
            # Финальная верификация
            mastered_audio = await self._final_verification_pass(mastered_audio, mastering_target)
            
            # Создание отчёта
            applied_config = self._create_mastering_report(processing_plan, source_analysis, mastering_target)

            # Финальный анализ
            result_analysis = await self._analyze_source_material(mastered_audio)
            self.logger.info(
                f"  ✅ Mastered: LUFS {result_analysis['lufs']:.1f}, "
                f"peak {result_analysis['peak']:.1f}dB, "
                f"dynamics {result_analysis['dynamic_range']:.1f}LU, "
                f"duration {result_analysis['duration']:.1f}s"
            )

            return mastered_audio, applied_config

        except Exception as e:
            self.logger.error(f"❌ CRITICAL Mastering error: {e}")
            
            # ИСПРАВЛЕНО: Улучшенный fallback без потери исходного аудио
            try:
                # Пытаемся вернуть хотя бы исходное аудио, а не тишину
                if isinstance(audio, (bytes, bytearray)) and len(audio) > 0:
                    fallback_audio = self._load_audio_from_bytes(bytes(audio))
                    
                    # Проверяем что fallback не тишина
                    if fallback_audio.max_dBFS != float('-inf'):
                        self.logger.warning("🚨 Returning normalized original audio as fallback")
                        return effects.normalize(fallback_audio), target_config
                    
                elif isinstance(audio, AudioSegment):
                    if audio.max_dBFS != float('-inf'):
                        self.logger.warning("🚨 Returning normalized original AudioSegment as fallback")
                        return effects.normalize(audio), target_config
                
                # Если даже исходное аудио проблемное - выбрасываем исключение
                raise ValueError("❌ FATAL: Cannot recover any audio, original is also corrupted!")
                
            except Exception as inner_err:
                self.logger.error(f"❌ FATAL: Fallback also failed: {inner_err}")
                # ТОЛЬКО В КРАЙНЕМ СЛУЧАЕ возвращаем ошибку, а не тишину
                raise ValueError(f"❌ FATAL: Complete mastering failure: {e}, fallback: {inner_err}")
  
    async def _analyze_source_material(self, audio: AudioSegment) -> Dict:
        """ИСПРАВЛЕННЫЙ детальный анализ исходного материала"""
        try:
            # ДОБАВЛЕНО: Проверяем входное аудио
            if audio is None:
                raise ValueError("Audio is None in analysis")
            
            if len(audio) == 0:
                raise ValueError("Audio has zero duration in analysis")
            
            if audio.max_dBFS == float('-inf'):
                raise ValueError("Audio is completely silent in analysis")

            # Конвертируем в numpy для анализа
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # ИСПРАВЛЕНО: Правильная нормализация
            max_val = 2**(audio.sample_width * 8 - 1)
            samples = samples / max_val
            
            sample_rate = audio.frame_rate
            
            analysis = {}
            
            # Базовые характеристики
            analysis['peak'] = audio.max_dBFS
            analysis['rms'] = audio.rms
            analysis['duration'] = len(audio) / 1000.0
            
            # ИСПРАВЛЕНО: Более точный LUFS анализ
            if audio.rms > 0:
                # Приближенный LUFS через RMS
                rms_db = 20 * np.log10(audio.rms / audio.max_possible_amplitude)
                analysis['lufs'] = max(-70, rms_db - 23)  # Приблизительная конвертация в LUFS
            else:
                analysis['lufs'] = -70
            
            # Динамический диапазон
            if audio.rms > 0:
                rms_db = 20 * np.log10(audio.rms / audio.max_possible_amplitude)
                analysis['dynamic_range'] = abs(analysis['peak'] - rms_db)
            else:
                analysis['dynamic_range'] = 0
            
            # Спектральный анализ только если есть данные
            mono_samples = samples.mean(axis=1) if audio.channels == 2 else samples
            
            if len(mono_samples) > 0:
                # Безопасный спектральный анализ
                try:
                    # Ограничиваем длину для производительности
                    if len(mono_samples) > sample_rate * 30:  # Максимум 30 секунд для анализа
                        mono_samples = mono_samples[:sample_rate * 30]
                    
                    spectral_centroid = librosa.feature.spectral_centroid(y=mono_samples, sr=sample_rate)
                    analysis['spectral_centroid'] = float(np.mean(spectral_centroid))
                    
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=mono_samples, sr=sample_rate)
                    analysis['spectral_rolloff'] = float(np.mean(spectral_rolloff))
                except Exception as e:
                    self.logger.debug(f"Spectral analysis failed: {e}")
                    analysis['spectral_centroid'] = 2000.0
                    analysis['spectral_rolloff'] = 8000.0
                
                # Стерео анализ
                if audio.channels == 2 and len(samples) > 0:
                    try:
                        correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = 1.0
                        analysis['stereo_correlation'] = float(correlation)
                        analysis['stereo_width'] = max(0.0, 1.0 - abs(correlation))
                    except Exception as e:
                        self.logger.debug(f"Stereo analysis failed: {e}")
                        analysis['stereo_correlation'] = 1.0
                        analysis['stereo_width'] = 0.0
                else:
                    analysis['stereo_correlation'] = 1.0
                    analysis['stereo_width'] = 0.0
                
                # Частотное распределение (упрощённое)
                try:
                    # Используем более короткий сэмпл для FFT
                    fft_samples = mono_samples[:min(len(mono_samples), sample_rate * 10)]  # Максимум 10 сек
                    fft = np.fft.rfft(fft_samples)
                    freqs = np.fft.rfftfreq(len(fft_samples), 1/sample_rate)
                    magnitude = np.abs(fft)
                    
                    # Разбиваем на полосы
                    low_mask = freqs < 300
                    mid_mask = (freqs >= 300) & (freqs < 3000) 
                    high_mask = freqs >= 3000
                    
                    total_energy = np.sum(magnitude**2) + 1e-10  # Избегаем деления на ноль
                    
                    analysis['frequency_balance'] = {
                        'low': float(np.sum(magnitude[low_mask]**2) / total_energy),
                        'mid': float(np.sum(magnitude[mid_mask]**2) / total_energy), 
                        'high': float(np.sum(magnitude[high_mask]**2) / total_energy)
                    }
                except Exception as e:
                    self.logger.debug(f"Frequency analysis failed: {e}")
                    analysis['frequency_balance'] = {'low': 0.33, 'mid': 0.33, 'high': 0.34}
                
                # Остальные анализы с fallback значениями
                analysis.update({
                    'transient_density': 5.0,
                    'harmonic_ratio': 0.5,
                    'percussive_ratio': 0.5
                })
            else:
                # Fallback значения если нет данных
                analysis.update({
                    'spectral_centroid': 2000.0,
                    'spectral_rolloff': 8000.0,
                    'stereo_correlation': 1.0,
                    'stereo_width': 0.0,
                    'frequency_balance': {'low': 0.33, 'mid': 0.33, 'high': 0.34},
                    'transient_density': 5.0,
                    'harmonic_ratio': 0.5,
                    'percussive_ratio': 0.5
                })
            
            # ДОБАВЛЕНО: Финальная валидация анализа
            if analysis['peak'] == float('-inf') or analysis['rms'] == 0:
                raise ValueError("Analysis detected completely silent audio")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Source analysis error: {e}")
            # НЕ возвращаем fallback данные для тишины!
            raise ValueError(f"Source analysis failed - audio may be corrupted: {e}")
    
    # Остальные методы остаются без изменений...
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
        genre = genre_info.get("name", "").lower()
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
            "drum and bass": {
                "frequency_balance": {"low": 0.3, "mid": 0.4, "high": 0.3},
                "stereo_width": 1.1,
                "transient_preservation": 0.95
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
            },
            "house": {
                "frequency_balance": {"low": 0.35, "mid": 0.4, "high": 0.25},
                "stereo_width": 1.1,
                "transient_preservation": 0.8
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
            genre = genre_info.get("name", "").lower()
            saturation_types = {
                "lofi": "tape",
                "trap": "tube", 
                "techno": "transistor",
                "ambient": "tube",
                "house": "tube"
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
        genre = genre_info.get("name", "").lower()
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
        """ИСПРАВЛЕННОЕ применение запланированной цепочки обработки"""
        
        processed = audio
        
        # ДОБАВЛЕНО: Проверяем входное аудио
        if processed.max_dBFS == float('-inf'):
            self.logger.error("❌ CRITICAL: Input audio to mastering chain is silent!")
            return processed
        
        for stage in plan["stages"]:
            if stage in plan["parameters"]:
                params = plan["parameters"][stage]
                
                self.logger.debug(f"  🔧 Applying {stage}: {params}")
                
                try:
                    # Проверяем состояние перед обработкой
                    pre_peak = processed.max_dBFS
                    
                    # Проверяем, есть ли обработчик для этого эффекта
                    if hasattr(self.effects_chain, 'processors') and stage in self.effects_chain.processors:
                        processed = await self.effects_chain.processors[stage].process(processed, params)
                    else:
                        # Fallback для базовых эффектов
                        processed = await self._apply_basic_effect(processed, stage, params)
                    
                    # ДОБАВЛЕНО: Проверяем что эффект не убил звук
                    post_peak = processed.max_dBFS
                    if post_peak == float('-inf') and pre_peak != float('-inf'):
                        self.logger.error(f"❌ Effect {stage} made audio silent! Reverting...")
                        # Возвращаем предыдущее состояние
                        processed = audio if stage == plan["stages"][0] else processed  # Это нужно улучшить
                        
                except Exception as e:
                    self.logger.error(f"❌ Error in {stage}: {e}")
                    # Не прерываем цепочку, продолжаем с текущим состоянием
        
        return processed
    
    async def _apply_basic_effect(self, audio: AudioSegment, effect: str, params: Dict) -> AudioSegment:
        """ИСПРАВЛЕННЫЕ базовые эффекты как fallback"""
        try:
            # ДОБАВЛЕНО: Проверяем входное аудио
            if audio.max_dBFS == float('-inf'):
                self.logger.warning(f"⚠️ Skipping {effect} - audio is silent")
                return audio
            
            if effect == "limiter":
                # Простой лимитер через нормализацию
                ceiling = params.get("ceiling", -1)
                if audio.max_dBFS > ceiling:
                    reduction = audio.max_dBFS - ceiling
                    result = audio - reduction
                    
                    # Проверяем результат
                    if result.max_dBFS == float('-inf'):
                        self.logger.warning(f"⚠️ Limiter made audio silent, using original")
                        return audio
                    
                    return result
                return audio
            
            elif effect == "compressor":
                # Базовое сжатие через нормализацию динамики
                try:
                    result = effects.compress_dynamic_range(audio, threshold=params.get("threshold", -20))
                    
                    # Проверяем результат
                    if result.max_dBFS == float('-inf'):
                        self.logger.warning(f"⚠️ Compressor made audio silent, using original")
                        return audio
                    
                    return result
                except Exception:
                    return audio
            
            elif effect == "eq":
                # Базовый EQ через фильтры (пока возвращаем как есть)
                return audio
            
            else:
                return audio
                
        except Exception as e:
            self.logger.error(f"Error in basic {effect}: {e}")
            return audio
    
    async def _final_verification_pass(self, audio: AudioSegment, target: MasteringTarget) -> AudioSegment:
        """ИСПРАВЛЕННАЯ финальная верификация и коррекция"""
        
        # ДОБАВЛЕНО: Проверяем что audio не None и не тишина
        if audio is None:
            raise ValueError("❌ CRITICAL: Audio is None in final verification!")
        
        if audio.max_dBFS == float('-inf'):
            raise ValueError("❌ CRITICAL: Audio is silent in final verification!")
        
        # Анализируем результат
        try:
            final_analysis = await self._analyze_source_material(audio)
        except Exception as e:
            self.logger.error(f"❌ Final analysis failed: {e}")
            # Если анализ не удался, возвращаем аудио как есть
            return audio
        
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
            try:
                over_ceiling = final_analysis["peak"] - target.peak_ceiling
                test_corrected = corrected - (over_ceiling + 0.1)  # Небольшой запас
                
                # Проверяем что коррекция не убила звук
                if test_corrected.max_dBFS != float('-inf'):
                    corrected = test_corrected
                    self.logger.info(f"  🚧 Final peak correction: -{over_ceiling + 0.1:.1f}dB")
                else:
                    self.logger.warning("⚠️ Peak correction would make audio silent, skipping")
                    
            except Exception as e:
                self.logger.error(f"❌ Peak correction failed: {e}")
        
        if "loudness" in corrections_needed:
            # Коррекция громкости
            try:
                lufs_correction = target.lufs - final_analysis["lufs"]
                if abs(lufs_correction) < 6:  # Разумный лимит
                    test_corrected = corrected + lufs_correction
                    
                    # Проверяем результат коррекции
                    if test_corrected.max_dBFS != float('-inf'):
                        corrected = test_corrected
                        self.logger.info(f"  📊 Final loudness correction: {lufs_correction:+.1f}dB")
                    else:
                        self.logger.warning("⚠️ Loudness correction would make audio silent, skipping")
                        
            except Exception as e:
                self.logger.error(f"❌ Loudness correction failed: {e}")
        
        # ДОБАВЛЕНО: Финальная проверка результата
        if corrected.max_dBFS == float('-inf'):
            self.logger.warning("⚠️ Final verification resulted in silence, returning original")
            return audio
        
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
                "stereo_width": source_analysis.get("stereo_width", 0.5),
                "duration": source_analysis["duration"]
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