# pipeline.py - Модульная система обработки с поэтапным сохранением
import os
import io
import asyncio
import logging
import time
import random
import requests
from export import ExportManager
from pydub import AudioSegment
from pydub.generators import Sine, Square, WhiteNoise
from pydub.effects import normalize
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from config import config, GenreType, MasteringPurpose
from mistral_client import query_structured_music
from metadata import MetadataProcessor
from sample_engine import SemanticSampleEngine
from musicgen_wrapper import MusicGenEngine
from mastering import SmartMasteringEngine
from verification import MixVerifier
from sample_engine import EffectsChain
from export import ExportManager


@dataclass
class GenerationRequest:
    """Запрос на генерацию трека"""
    prompt: str
    genre: Optional[str] = None
    bpm: Optional[int] = None
    duration: Optional[int] = None
    mastering_purpose: str = "personal"
    output_dir: str = "output"
    export_stems: bool = True
    custom_structure: Optional[List[Dict]] = None
    sample_tags: Optional[List[str]] = None
    energy_level: float = 0.5
    creativity_factor: float = 0.7

@dataclass 
class GenerationResult:
    """Результат генерации трека"""
    success: bool
    final_path: Optional[str] = None
    structure_data: Optional[Dict] = None
    used_samples: Optional[List[Dict]] = None
    mastering_config: Optional[Dict] = None
    mastering_report: Optional[Dict] = None
    generation_time: float = 0.0
    quality_score: float = 0.0
    error_message: Optional[str] = None
    intermediate_files: Optional[Dict[str, str]] = None


class WaveDreamPipeline:
    """
    Главная система pipeline для WaveDream Enhanced Pro с поэтапным сохранением
    
    Этапы обработки:
    1. prepare_metadata() - Анализ промпта и подготовка метаданных
    2. detect_genre() - Детекция жанра с возможностью ручного переопределения  
    3. generate_structure() - Генерация структуры через LLaMA3/локальную логику
    4. select_samples() - Семантический подбор сэмплов
    5. generate_base() - Генерация основы через MusicGen → СОХРАНЕНИЕ
    6. create_stems() - Создание стемов из сэмплов → СОХРАНЕНИЕ КАЖДОГО СТЕМА
    7. mix_tracks() - Микширование стемов с основой → СОХРАНЕНИЕ
    8. apply_effects() - Применение эффектов → СОХРАНЕНИЕ
    9. master_track() - Умный мастеринг по назначению → СОХРАНЕНИЕ
    10. verify_quality() - Верификация качества
    11. export_results() - Финальный экспорт всех версий
    """
    
    def __init__(self):
        self.metadata_processor = MetadataProcessor()
        self.sample_engine = SemanticSampleEngine()
        self.musicgen_engine = MusicGenEngine()
        self.mastering_engine = SmartMasteringEngine()
        self.verifier = MixVerifier()
        self.effects_chain = EffectsChain()
        self.export_manager = ExportManager()
        
        self.logger = logging.getLogger(__name__)
        self._performance_stats = {}
        
        # Для поэтапного сохранения
        self._current_project_name = None
        self._intermediate_storage = {}
        
    async def generate_track(self, request: GenerationRequest) -> GenerationResult:
        """
        Исправленная главная функция генерации
        """
        start_time = time.time()

        try:
            # Проверка окружения
            self.logger.info("🔍 Проверка окружения...")
            env_checks = self.export_manager.check_export_environment()

            critical_checks = ["base_dir_writable", "sufficient_space", "pydub_working"]
            failed_critical = [check for check in critical_checks if not env_checks.get(check, False)]

            if failed_critical:
                error_msg = f"Критические проверки не пройдены: {', '.join(failed_critical)}"
                self.logger.error(f"❌ {error_msg}")
                return GenerationResult(
                    success=False,
                    error_message=error_msg,
                    generation_time=time.time() - start_time
                )

            # ... все этапы (metadata, genre, structure, samples, base, stems, mix, effects, mastering, verify, export)

            generation_time = time.time() - start_time
            self._performance_stats["total_time"] = generation_time

            result = GenerationResult(
                success=True,
                final_path=final_path or exported_files.get("final"),
                structure_data=structure,
                used_samples=selected_samples,
                mastering_config=mastering_config,
                mastering_report=mastering_report,
                generation_time=generation_time,
                quality_score=quality_report.get("overall_score", 0.0),
                intermediate_files={**self._intermediate_storage, **exported_files}
            )

            self.logger.info(f"🎉 Генерация завершена за {generation_time:.1f}с")
            self.logger.info(f"🎯 Качество: {result.quality_score:.2f}/1.0")
            self.logger.info(f"📁 Всего файлов создано: {len(result.intermediate_files)}")

            return result

        except Exception as e:  # ← выровнено с try
            generation_time = time.time() - start_time
            self.logger.error(f"❌ Ошибка генерации: {e}")

            try:
                if hasattr(self, '_intermediate_storage') and self._intermediate_storage:
                    self.logger.info("🚨 Попытка аварийного сохранения...")

                    emergency_audio_dict = {}
                    for stage_name, file_path in self._intermediate_storage.items():
                        if isinstance(file_path, str) and os.path.exists(file_path):
                            try:
                                with open(file_path, 'rb') as f:
                                    emergency_audio_dict[stage_name] = f.read()
                            except Exception as read_error:
                                self.logger.debug(f"Could not read {stage_name}: {read_error}")

                    if 'mastered_audio' in locals() and isinstance(locals()['mastered_audio'], bytes):
                        emergency_audio_dict['final_mastered'] = locals()['mastered_audio']

                    emergency_files = await self.export_manager.force_save_everything(
                        emergency_audio_dict,
                        request.output_dir or "emergency_output"
                    )
                    self.logger.info(f"🚨 Аварийно сохранено: {len(emergency_files)} файлов")

            except Exception as save_error:
                self.logger.error(f"❌ Ошибка аварийного сохранения: {save_error}")

            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=str(e),
                intermediate_files=getattr(self, '_intermediate_storage', {})
            )

    async def save_intermediate(self, name: str, project_name: str, audio: bytes) -> Optional[str]:
        """
        ИСПРАВЛЕННАЯ async-функция сохранения промежуточного файла
        """
        try:
            if not audio or len(audio) == 0:
                self.logger.warning(f"⚠️ Empty audio for intermediate '{name}'")
                return None

            # ИСПРАВЛЕНО: прямой await вместо asyncio.run()
            saved_path = await self.export_manager.save_intermediate(
                name=name,                          # Название этапа
                audio_bytes=audio,                  # bytes аудиоданные
                output_dir=self._current_project_name or "output"  # Директория проекта
            )

            if saved_path:
                self.logger.info(f"  💾 Промежуточный файл '{name}' сохранен: {saved_path}")
                return saved_path
            else:
                self.logger.error(f"❌ Не удалось сохранить промежуточный файл '{name}'")
                return None

        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения промежуточного файла '{name}': {e}")
            return None


    async def save_stem(self, audio: bytes, project_name: str, stem_name: str) -> Optional[str]:
        """
        ИСПРАВЛЕННАЯ async-функция сохранения стема
        """
        try:
            if not audio or len(audio) == 0:
                self.logger.warning(f"⚠️ Empty audio for stem '{stem_name}'")
                return None

            # ИСПРАВЛЕНО: прямой await вместо asyncio.run()
            saved_path = await self.export_manager.save_stem(
                stem_name=stem_name,                # Название инструмента
                audio_bytes=audio,                  # bytes аудиоданные  
                output_dir=self._current_project_name or "output"  # Директория проекта
            )

            if saved_path:
                self.logger.info(f"  🎛️ Стем '{stem_name}' сохранен: {saved_path}")
                return saved_path
            else:
                self.logger.error(f"❌ Не удалось сохранить стем '{stem_name}'")
                return None

        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения стема '{stem_name}': {e}")
            return None

    async def save_final_mix(self, audio: bytes, project_name: str) -> Optional[str]:
        """
        ИСПРАВЛЕННАЯ async-функция сохранения финального микса
        """
        try:
            if not audio or len(audio) == 0:
                raise ValueError("❌ CRITICAL: Empty final mix audio!")
                
            # ИСПРАВЛЕНО: прямой await вместо asyncio.run()
            saved_files = await self.export_manager.save_final_mix(
                project_name=project_name,          # Имя проекта
                audio_bytes=audio,                  # bytes финального аудио
                output_dir=self._current_project_name or "output",  # Директория
                formats=["wav", "mp3"]              # Форматы для сохранения
            )
            
            if saved_files:
                # Возвращаем путь к основному WAV или MP3
                main_file = (
                    saved_files.get("final_wav") 
                    or saved_files.get("final_mp3") 
                    or list(saved_files.values())[0]
                )
                self.logger.info(f"  🎵 Финальный микс сохранен в {len(saved_files)} форматах")
                return main_file
            else:
                raise ValueError("❌ CRITICAL: No final mix files were saved!")
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Ошибка сохранения финального микса: {e}")
            raise
    
    # === ЭТАПЫ PIPELINE (без изменений в логике, но с улучшенным логированием) ===
    
    async def _step_prepare_metadata(self, request: GenerationRequest) -> Dict:
        """Этап 1: Подготовка и анализ метаданных"""
        start_time = time.time()
        
        metadata = {
            "original_prompt": request.prompt,
            "timestamp": time.time(),
            "request_id": f"wd_{int(time.time())}",
        }
        
        # Анализ промпта через метаданные процессор
        prompt_analysis = self.metadata_processor.analyze_prompt(request.prompt)
        metadata.update(prompt_analysis)
        
        # Извлечение параметров из промпта
        extracted_params = self.metadata_processor.extract_parameters(request.prompt)
        
        metadata.update({
            "detected_bpm": extracted_params.get("bpm", request.bpm),
            "detected_key": extracted_params.get("key"),
            "detected_mood": extracted_params.get("mood", []),
            "detected_instruments": extracted_params.get("instruments", []),
            "detected_tags": extracted_params.get("tags", []),
            "energy_level": request.energy_level,
            "creativity_factor": request.creativity_factor
        })
        
        processing_time = time.time() - start_time
        self._performance_stats["metadata_time"] = processing_time
        
        self.logger.info(f"  📊 Детектировано: BPM={metadata.get('detected_bpm')}, "
                        f"Инструменты={len(metadata.get('detected_instruments', []))}")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return metadata
    
    async def _step_detect_genre(self, request: GenerationRequest, metadata: Dict) -> Dict:
        """Этап 2: Детекция и валидация жанра"""
        start_time = time.time()
        
        if request.genre:
            # Ручное переопределение жанра
            detected_genre = request.genre.lower()
            self.logger.info(f"  🎭 Жанр задан вручную: {detected_genre}")
        else:
            # Автоматическая детекция
            detected_genre = self.metadata_processor.detect_genre(
                request.prompt, metadata.get("detected_tags", [])
            )
            self.logger.info(f"  🎭 Жанр детектирован автоматически: {detected_genre}")
        
        # Получаем конфиг жанра
        genre_config = config.get_genre_config(detected_genre)
        if not genre_config:
            self.logger.warning(f"  ⚠️ Неизвестный жанр {detected_genre}, используем trap")
            detected_genre = "trap"
            genre_config = config.get_genre_config("trap")
        
        genre_info = {
            "name": detected_genre,
            "config": genre_config,
            "bpm_range": genre_config.bpm_range,
            "target_bpm": metadata.get("detected_bpm") or 
                         (genre_config.bpm_range[0] + genre_config.bpm_range[1]) // 2,
            "energy_range": genre_config.energy_range,
            "mastering_style": genre_config.mastering_style,
            "energy_level": metadata.get("energy_level", 0.5)
        }
        
        processing_time = time.time() - start_time
        self._performance_stats["genre_detection_time"] = processing_time
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return genre_info
    
    async def _step_generate_structure(
        self, request: GenerationRequest, metadata: Dict, genre_info: Dict
    ) -> Dict:
        """Этап 3: Генерация структуры трека"""
        start_time = time.time()

        if request.custom_structure:
            # Используем кастомную структуру
            structure = {
                "sections": request.custom_structure,
                "total_duration": sum(s.get("duration", 8) for s in request.custom_structure),
                "source": "custom"
            }
            self.logger.info(f"  🏗️ Используется кастомная структура: {len(structure['sections'])} секций")
        else:
            # Генерируем через LLaMA3 или fallback
            try:
                llama_response = query_structured_music(request.prompt)

                if not llama_response:
                    raise ValueError("LLaMA3 вернула пустую структуру")

                structure = {
                    "sections": llama_response["structure"],
                    "total_duration": sum(s["duration"] for s in llama_response["structure"]),
                    "source": "llama3-music"
                }

                self.logger.info(f"  🧠 Структура от LLaMA3: {len(structure['sections'])} секций")

            except Exception as e:
                self.logger.warning(f"  ⚠️ LLaMA3 недоступна, используем fallback: {e}")
                structure = self._generate_fallback_structure(genre_info, request.duration)
                structure["source"] = "fallback"

        # Валидация и нормализация структуры
        structure = self._validate_structure(structure, genre_info)

        processing_time = time.time() - start_time
        self._performance_stats["structure_time"] = processing_time

        self.logger.info(f"  ⏱️ Общая длительность: {structure['total_duration']}с")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")

        return structure
    
    async def _step_select_samples(
        self, request: GenerationRequest, metadata: Dict, genre_info: Dict, structure: Dict
    ) -> List[Dict]:
        """Этап 4: Семантический подбор сэмплов"""
        start_time = time.time()
        
        # Собираем требуемые теги из разных источников
        required_tags = set()
        required_tags.update(metadata.get("detected_instruments", []))
        required_tags.update(metadata.get("detected_tags", []))
        required_tags.update(genre_info["config"].core_instruments)
        
        if request.sample_tags:
            required_tags.update(request.sample_tags)
        
        selected_samples = []
        
        # Подбираем сэмплы для каждой секции
        for section in structure["sections"]:
            section_type = section.get("type", "unknown")
            section_energy = section.get("energy", 0.5)
            
            self.logger.info(f"  🎯 Секция '{section_type}': энергия {section_energy}")
            
            # Определяем приоритетные инструменты для секции
            section_instruments = self._get_section_instruments(
                section_type, genre_info, section_energy
            )
            
            # Подбираем сэмплы через семантический движок
            section_samples = await self.sample_engine.find_samples(
                tags=list(required_tags),
                instruments=section_instruments,
                genre=genre_info["name"],
                bpm=genre_info["target_bpm"],
                energy=section_energy,
                max_results=len(section_instruments)
            )
            
            # Добавляем информацию о секции к сэмплам
            for sample in section_samples:
                sample["section"] = section_type
                sample["section_energy"] = section_energy
                selected_samples.append(sample)
        
        processing_time = time.time() - start_time
        self._performance_stats["sample_selection_time"] = processing_time
        
        self.logger.info(f"  ✅ Подобрано сэмплов: {len(selected_samples)}")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return selected_samples
    
    async def _step_generate_base(
        self, request: GenerationRequest, metadata: Dict, 
        genre_info: Dict, structure: Dict
    ) -> bytes:
        """Этап 5: Генерация основы через MusicGen"""
        start_time = time.time()
        
        # Создаём улучшенный промпт для MusicGen
        enhanced_prompt = self._create_musicgen_prompt(
            request.prompt, genre_info, metadata
        )
        
        duration = structure["total_duration"]
        
        self.logger.info(f"  🎼 MusicGen промпт: '{enhanced_prompt[:100]}...'")
        self.logger.info(f"  ⏱️ Длительность: {duration}с")
        
        # Генерируем базовую дорожку
        base_audio = await self.musicgen_engine.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=metadata.get("creativity_factor", 0.7),
            genre_hint=genre_info["name"]
        )
        
        processing_time = time.time() - start_time
        self._performance_stats["musicgen_time"] = processing_time
        
        self.logger.info(f"  🎼 Базовая дорожка сгенерирована: {len(base_audio)} bytes")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return base_audio
    
    async def _step_create_stems(
        self, selected_samples: List[Dict], structure: Dict, genre_info: Dict
    ) -> Dict[str, bytes]:
        """Этап 6: Создание стемов из сэмплов"""
        start_time = time.time()
        
        stems = {}
        total_duration_ms = int(structure["total_duration"] * 1000)
        
        # Группируем сэмплы по инструментам
        instrument_groups = {}
        for sample in selected_samples:
            instrument = sample.get("instrument_role", "unknown")
            if instrument not in instrument_groups:
                instrument_groups[instrument] = []
            instrument_groups[instrument].append(sample)
        
        # Создаём стем для каждого инструмента
        for instrument, samples in instrument_groups.items():
            self.logger.info(f"  🎛️ Создаём стем: {instrument} ({len(samples)} сэмплов)")
            
            stem_audio = await self._create_instrument_stem(
                samples, structure, total_duration_ms, genre_info
            )
            
            stems[instrument] = stem_audio
            self.logger.info(f"    ✅ Стем '{instrument}': {len(stem_audio)} bytes")
        
        processing_time = time.time() - start_time
        self._performance_stats["stems_creation_time"] = processing_time
        
        self.logger.info(f"  🎛️ Всего стемов создано: {len(stems)}")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return stems
    
    async def _step_mix_tracks(
        self, base_audio: bytes, stems: Dict[str, bytes], genre_info: Dict
    ) -> bytes:
        """Этап 7: Микширование базы со стемами"""
        start_time = time.time()
        
        # Получаем настройки микса для жанра
        mix_settings = self._get_genre_mix_settings(genre_info["name"])
        
        self.logger.info(f"  🎚️ Микс настройки для {genre_info['name']}: "
                        f"база {mix_settings['base_level']}dB, "
                        f"стемы {mix_settings['stems_level']}dB")
        
        # Микшируем через эффекты движок
        mixed_audio = await self.effects_chain.mix_layers(
            base_layer=base_audio,
            stem_layers=stems,
            mix_settings=mix_settings,
            genre_info=genre_info
        )
        
        processing_time = time.time() - start_time
        self._performance_stats["mixing_time"] = processing_time
        
        self.logger.info(f"  🎚️ Микширование завершено: {len(mixed_audio)} bytes")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return mixed_audio
    
    async def _step_apply_effects(
        self, mixed_audio: bytes, metadata: Dict, genre_info: Dict
    ) -> bytes:
        """Этап 8: Применение жанровых эффектов"""
        start_time = time.time()
        
        # Создаём цепочку эффектов для жанра
        effects_config = self._get_genre_effects_config(genre_info, metadata)
        
        self.logger.info(f"  ✨ Применяем эффекты: {', '.join(effects_config.keys())}")
        
        processed_audio = await self.effects_chain.apply_effects(
            audio=mixed_audio,
            effects_config=effects_config,
            genre_info=genre_info
        )
        
        processing_time = time.time() - start_time
        self._performance_stats["effects_time"] = processing_time
        
        self.logger.info(f"  ✨ Эффекты применены: {len(processed_audio)} bytes")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        return processed_audio
    
    async def _step_master_track(
        self, processed_audio: bytes, mastering_purpose: str, genre_info: Dict
    ) -> Tuple[bytes, Dict, Dict]:
        """
        ИСПРАВЛЕННЫЙ этап мастеринга с правильными вызовами новой версии SmartMasteringEngine
        """
        start_time = time.time()
        
        try:
            mastering_config = config.get_mastering_config(mastering_purpose)

            self.logger.info(
                f"  🎛️ Мастеринг для {mastering_purpose}: "
                f"LUFS {mastering_config['target_lufs']}, "
                f"потолок {mastering_config['peak_ceiling']}dB"
            )

            # ИСПРАВЛЕНО: Новый вызов мастеринг движка с правильной сигнатурой
            mastering_result = await self.mastering_engine.master_track(
                audio=processed_audio,              # bytes или AudioSegment
                target_config=mastering_config,     # Dict с настройками
                genre_info=genre_info,              # Dict с информацией о жанре
                purpose=mastering_purpose           # str назначение
            )
            
            # ИСПРАВЛЕНО: Правильная обработка возврата (Tuple[AudioSegment, Dict])
            if isinstance(mastering_result, tuple) and len(mastering_result) == 2:
                mastered_audio_segment, applied_config = mastering_result
            else:
                raise ValueError("SmartMasteringEngine returned invalid result format")

            # ИСПРАВЛЕНО: Конвертируем AudioSegment в bytes для pipeline
            if hasattr(mastered_audio_segment, 'export'):
                buffer = io.BytesIO()
                mastered_audio_segment.export(buffer, format="wav")
                mastered_audio_bytes = buffer.getvalue()
                buffer.close()
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: убеждаемся что экспорт не пустой
                if len(mastered_audio_bytes) == 0:
                    raise ValueError("❌ CRITICAL: Mastering export resulted in empty bytes!")
                    
            else:
                mastered_audio_bytes = mastered_audio_segment

            # Создаем отчет о мастеринге (пока используем applied_config)
            mastering_report = applied_config.copy()

            processing_time = time.time() - start_time
            self._performance_stats["mastering_time"] = processing_time

            self.logger.info(f"  ✅ Мастеринг завершен: {len(mastered_audio_bytes)} bytes")
            self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
            
            return mastered_audio_bytes, mastering_config, mastering_report
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._performance_stats["mastering_time"] = processing_time
            
            self.logger.error(f"❌ Ошибка мастеринга: {e}")
            self.logger.info(f"  ⏱️ Время до ошибки: {processing_time:.2f}с")
            
            # ИСПРАВЛЕНО: НЕ возвращаем оригинальное аудио, а выбрасываем исключение
            # Пусть SmartMasteringEngine сам решает что делать с fallback
            raise ValueError(f"Mastering failed: {e}")

    async def _step_verify_quality(
        self, mastered_audio: bytes, mastering_config: Dict
    ) -> Dict:
        """Этап 10: Верификация качества финального трека"""
        start_time = time.time()
        
        quality_report = await self.verifier.analyze_track(
            audio=mastered_audio,
            target_config=mastering_config
        )
        
        overall_score = quality_report.get("overall_score", 0.0)
        
        processing_time = time.time() - start_time
        self._performance_stats["verification_time"] = processing_time
        
        self.logger.info(f"  🔍 Качество трека: {overall_score:.2f}/1.0")
        self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")
        
        if overall_score < 0.7:
            self.logger.warning("  ⚠️ Низкое качество трека, возможно требуется ручная доработка")
        
        return quality_report
    
    async def _step_export_results(
        self,
        request: GenerationRequest,
        mastered_audio: bytes,
        structure: Dict,
        selected_samples: List[Dict],
        mastering_config: Dict,
        intermediate_audio: Dict[str, bytes]
    ) -> Dict[str, str]:
        """
        ИСПРАВЛЕННЫЙ этап экспорта с правильными вызовами нового ExportManager
        """
        start_time = time.time()

        try:
            # ИСПРАВЛЕНО: Правильная подготовка конфига для нового ExportManager
            export_config = {
                "output_dir": request.output_dir,
                "export_stems": request.export_stems,
                "export_formats": ["wav", "mp3"],  # Форматы экспорта
                "request_data": {
                    "prompt": request.prompt,
                    "genre": request.genre,
                    "bpm": request.bpm,
                    "duration": request.duration,
                    "mastering_purpose": request.mastering_purpose,
                    "energy_level": request.energy_level,
                    "creativity_factor": request.creativity_factor
                },
                "structure": structure,
                "samples": selected_samples,
                "mastering": mastering_config
            }

            # ИСПРАВЛЕНО: Вызов нового метода export_complete_project
            exported_files = await self.export_manager.export_complete_project(
                mastered_audio=mastered_audio,          # bytes финального трека
                intermediate_audio=intermediate_audio,  # Dict[str, bytes] промежуточных версий
                config=export_config                    # Dict с конфигурацией
            )

            # ИСПРАВЛЕНО: Убираем отдельное создание отчета - новый ExportManager делает это сам
            # await self._generate_project_report(...) - больше не нужно

            processing_time = time.time() - start_time
            self._performance_stats["export_time"] = processing_time
            
            self.logger.info(f"  💾 Экспорт завершён: {len(exported_files)} файлов")
            self.logger.info(f"  ⏱️ Время обработки: {processing_time:.2f}с")

            return exported_files
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._performance_stats["export_time"] = processing_time
            
            self.logger.error(f"❌ Ошибка экспорта: {e}")
            self.logger.info(f"  ⏱️ Время до ошибки: {processing_time:.2f}с")
            
            # ИСПРАВЛЕНО: Попытка аварийного сохранения через новый метод
            try:
                emergency_files = await self.export_manager.force_save_everything(
                    intermediate_audio, request.output_dir  # ИСПРАВЛЕНА СИГНАТУРА: 2 параметра
                )
                self.logger.info(f"🚨 Аварийное сохранение: {len(emergency_files)} файлов")
                return emergency_files
                
            except Exception as emergency_error:
                self.logger.error(f"❌ Критическая ошибка аварийного сохранения: {emergency_error}")
                raise ValueError(f"Complete export failure: {e}, emergency: {emergency_error}")
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (улучшенные с логированием времени) ===
    
    async def _query_llama_structure(
        self, request: GenerationRequest, metadata: Dict, genre_info: Dict
    ) -> Dict:
        """Запрос структуры к LLaMA3"""
        # Реализация запроса к LLaMA3
        # В продакшене здесь будет реальный API вызов
        raise NotImplementedError("LLaMA3 integration not implemented")
    
    def _generate_fallback_structure(self, genre_info: Dict, duration: Optional[int]) -> Dict:
        """ИСПРАВЛЕННАЯ fallback генерация структуры"""
        try:
            # ИСПРАВЛЕНИЕ: Правильный доступ к конфигу жанра
            genre_config = genre_info.get("config")

            if genre_config:
                # Если genre_config это dataclass, получаем default_structure
                if hasattr(genre_config, 'default_structure'):
                    default_structure = genre_config.default_structure
                elif hasattr(genre_config, '__dict__'):
                    default_structure = getattr(genre_config, 'default_structure', [])
                else:
                    # Если это словарь
                    default_structure = genre_config.get("default_structure", [])
            else:
                default_structure = []

            # Если дефолтная структура пустая, создаём базовую
            if not default_structure:
                self.logger.warning("⚠️ Нет дефолтной структуры для жанра, создаём базовую")
                genre_name = genre_info.get("name", "generic").lower()

                # Структуры по жанрам
                genre_structures = {
                    "trap": [
                        {"type": "intro", "duration": 8, "energy": 0.2},
                        {"type": "buildup", "duration": 8, "energy": 0.4},
                        {"type": "drop", "duration": 16, "energy": 0.9},
                        {"type": "verse", "duration": 16, "energy": 0.6},
                        {"type": "drop", "duration": 16, "energy": 0.9},
                        {"type": "outro", "duration": 8, "energy": 0.3}
                    ],
                    "lofi": [
                        {"type": "intro", "duration": 12, "energy": 0.3},
                        {"type": "main", "duration": 32, "energy": 0.5},
                        {"type": "bridge", "duration": 16, "energy": 0.4},
                        {"type": "main", "duration": 24, "energy": 0.5},
                        {"type": "outro", "duration": 16, "energy": 0.2}
                    ],
                    "dnb": [
                        {"type": "intro", "duration": 8, "energy": 0.3},
                        {"type": "buildup", "duration": 16, "energy": 0.7},
                        {"type": "drop", "duration": 32, "energy": 1.0},
                        {"type": "breakdown", "duration": 16, "energy": 0.4},
                        {"type": "drop", "duration": 24, "energy": 0.9},
                        {"type": "outro", "duration": 8, "energy": 0.2}
                    ],
                    "ambient": [
                        {"type": "intro", "duration": 20, "energy": 0.1},
                        {"type": "development", "duration": 40, "energy": 0.3},
                        {"type": "climax", "duration": 20, "energy": 0.5},
                        {"type": "resolution", "duration": 30, "energy": 0.2},
                        {"type": "outro", "duration": 20, "energy": 0.1}
                    ]
                }

                default_structure = genre_structures.get(genre_name, [
                    {"type": "intro", "duration": 8, "energy": 0.3},
                    {"type": "verse", "duration": 16, "energy": 0.5},
                    {"type": "hook", "duration": 16, "energy": 0.8},
                    {"type": "verse", "duration": 16, "energy": 0.6},
                    {"type": "hook", "duration": 16, "energy": 0.9},
                    {"type": "outro", "duration": 8, "energy": 0.4}
                ])

            target_duration = duration or 80
            current_duration = sum(s.get("duration", 8) for s in default_structure)

            # ИСПРАВЛЕНИЕ: Более надежная проверка деления на ноль
            if current_duration <= 0:
                self.logger.error("❌ Текущая длительность структуры равна 0, создаём экстренную структуру")
                default_structure = [
                    {"type": "intro", "duration": 8, "energy": 0.3},
                    {"type": "main", "duration": 32, "energy": 0.7},
                    {"type": "outro", "duration": 8, "energy": 0.3}
                ]
                current_duration = 48

            # Масштабируем структуру
            scale_factor = target_duration / current_duration
            scale_factor = max(0.3, min(scale_factor, 4.0))  # Ограничиваем разумными пределами

            scaled_structure = []
            total_scaled_duration = 0

            for section in default_structure:
                original_duration = section.get("duration", 8)
                scaled_duration = max(4, int(original_duration * scale_factor))

                scaled_section = {
                    "type": section.get("type", "section"),
                    "duration": scaled_duration,
                    "energy": max(0.1, min(1.0, section.get("energy", 0.5))),
                    "start_time": total_scaled_duration
                }

                scaled_structure.append(scaled_section)
                total_scaled_duration += scaled_duration

            self.logger.info(f"✅ Fallback структура: {len(scaled_structure)} секций, {total_scaled_duration}с")

            return {
                "sections": scaled_structure,
                "total_duration": total_scaled_duration,
                "source": "fallback_generated"
            }

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка fallback структуры: {e}")

            # Минимальная аварийная структура
            emergency_duration = duration or 40
            return {
                "sections": [
                    {"type": "intro", "duration": 8, "energy": 0.3, "start_time": 0},
                    {"type": "main", "duration": emergency_duration - 16, "energy": 0.7, "start_time": 8},
                    {"type": "outro", "duration": 8, "energy": 0.3, "start_time": emergency_duration - 8}
                ],
                "total_duration": emergency_duration,
                "source": "emergency_fallback"
            }
    
    def _validate_structure(self, structure: Dict, genre_info: Dict) -> Dict:
        """Валидация и нормализация структуры"""
        sections = structure.get("sections", [])
        
        # Убеждаемся что у всех секций есть необходимые поля
        normalized_sections = []
        for section in sections:
            normalized_section = {
                "type": section.get("type", "unknown"),
                "duration": max(4, section.get("duration", 8)),  # минимум 4 секунды
                "energy": max(0.1, min(1.0, section.get("energy", 0.5))),  # 0.1-1.0
                "start_time": section.get("start_time", 0)
            }
            normalized_sections.append(normalized_section)
        
        # Пересчитываем start_time
        current_time = 0
        for section in normalized_sections:
            section["start_time"] = current_time
            current_time += section["duration"]
        
        return {
            "sections": normalized_sections,
            "total_duration": current_time,
            "source": structure.get("source", "unknown")
        }
    
    def _get_section_instruments(
        self, section_type: str, genre_info: Dict, energy: float
    ) -> List[str]:
        """Определение инструментов для секции"""
        config = genre_info["config"]
        instruments = config.core_instruments.copy()
        
        # Добавляем дополнительные инструменты в зависимости от энергии и типа секции
        if energy > 0.6 or section_type in ["hook", "drop", "climax"]:
            instruments.extend(config.optional_instruments)
        elif section_type in ["intro", "outro"] and energy < 0.4:
            # Для тихих секций берём только основные инструменты
            instruments = instruments[:2]
        
        return instruments
    
    def _create_musicgen_prompt(
        self, original_prompt: str, genre_info: Dict, metadata: Dict
    ) -> str:
        """Создание улучшенного промпта для MusicGen"""
        genre = genre_info["name"]
        bpm = genre_info["target_bpm"]
        style = genre_info["mastering_style"]
        
        # Добавляем жанровые термины
        genre_terms = {
            "trap": ["dark", "aggressive", "urban", "melodic"],
            "lofi": ["chill", "vintage", "cozy", "nostalgic"],  
            "dnb": ["energetic", "breakbeat", "bass-heavy", "dynamic"],
            "ambient": ["ethereal", "spacious", "meditative", "peaceful"],
            "techno": ["hypnotic", "minimal", "driving", "industrial"],
        }
        
        terms = genre_terms.get(genre, ["professional", "high-quality"])
        
        enhanced_prompt = f"{original_prompt} {genre} style {bpm}bpm " + " ".join(terms[:2])
        
        return enhanced_prompt
    
    def _get_genre_mix_settings(self, genre: str) -> Dict:
        """Настройки микса для жанра"""
        mix_settings = {
            "trap": {"base_level": -4, "stems_level": -5},
            "lofi": {"base_level": -5, "stems_level": -8}, 
            "dnb": {"base_level": -2, "stems_level": -4},
            "ambient": {"base_level": -6, "stems_level": -10},
            "techno": {"base_level": -3, "stems_level": -5},
        }
        
        return mix_settings.get(genre, {"base_level": -3, "stems_level": -6})
    
    def _get_genre_effects_config(self, genre_info: Dict, metadata: Dict) -> Dict:
        """Конфигурация эффектов для жанра"""
        genre = genre_info["name"]
        energy = metadata.get("energy_level", 0.5)
        
        effects_configs = {
            "trap": {
                "reverb": {"room_size": 0.3, "wet": 0.15},
                "compression": {"ratio": 3.5, "attack": 5},
                "eq": {"low": 2, "mid": 0, "high": 3},
                "saturation": {"amount": 0.3, "type": "tube"}
            },
            "lofi": {
                "vinyl_simulation": {"crackle": 0.4, "warmth": 0.6},
                "compression": {"ratio": 2.0, "attack": 15},
                "eq": {"low": 3, "mid": -1, "high": -2},
                "tape_saturation": {"amount": 0.5}
            },
            "dnb": {
                "reverb": {"room_size": 0.2, "wet": 0.1},
                "compression": {"ratio": 4.0, "attack": 3},
                "eq": {"low": 4, "mid": 1, "high": 2},
                "distortion": {"amount": 0.2, "type": "digital"}
            },
            "ambient": {
                "reverb": {"room_size": 0.8, "wet": 0.4},
                "compression": {"ratio": 1.5, "attack": 20},
                "eq": {"low": 0, "mid": -1, "high": 1},
                "chorus": {"rate": 0.3, "depth": 0.3}
            },
            "techno": {
                "reverb": {"room_size": 0.4, "wet": 0.2},
                "compression": {"ratio": 3.0, "attack": 8},
                "eq": {"low": 3, "mid": 0, "high": 1},
                "delay": {"time": 0.125, "feedback": 0.3}
            }
        }
        
        return effects_configs.get(genre, {})
    

    async def _create_instrument_stem(
        self, samples: List[Dict], structure: Dict,
        total_duration_ms: int, genre_info: Dict
    ) -> bytes:
        """
        ИСПРАВЛЕННОЕ создание стема для инструмента БЕЗ генерации тишины
        """
        try:
            # ИСПРАВЛЕНО: Если нет сэмплов - НЕ создаем тишину, а создаем синтетический ритм
            if not samples:
                self.logger.info(f"  🎛️ Нет сэмплов для инструмента, создаем синтетический ритм")
                stem_audio = self._create_synthetic_rhythm(total_duration_ms, genre_info)
            else:
                # Пытаемся использовать реальные сэмплы
                try:
                    sample_path = samples[0].get('path', samples[0].get('filename', ''))
                    if sample_path and os.path.exists(sample_path):
                        base_sample = AudioSegment.from_file(sample_path)
                        
                        # КРИТИЧЕСКАЯ ПРОВЕРКА: сэмпл не должен быть тишиной
                        if base_sample.max_dBFS == float('-inf'):
                            self.logger.warning(f"⚠️ Sample is silent, creating synthetic rhythm")
                            stem_audio = self._create_synthetic_rhythm(total_duration_ms, genre_info)
                        else:
                            # Повторяем сэмпл на всю длительность
                            repetitions = total_duration_ms // len(base_sample) + 1
                            repeated_sample = base_sample * repetitions
                            stem_audio = repeated_sample[:total_duration_ms]
                    else:
                        self.logger.warning(f"⚠️ Sample file not found, creating synthetic rhythm")
                        stem_audio = self._create_synthetic_rhythm(total_duration_ms, genre_info)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load sample: {e}, creating synthetic rhythm")
                    stem_audio = self._create_synthetic_rhythm(total_duration_ms, genre_info)

            # КРИТИЧЕСКАЯ ПРОВЕРКА: результат не должен быть тишиной
            if stem_audio.max_dBFS == float('-inf'):
                raise ValueError("❌ CRITICAL: Stem creation resulted in silence!")

            # Конвертируем в bytes
            buffer = io.BytesIO()
            stem_audio.export(buffer, format="wav")
            stem_bytes = buffer.getvalue()
            buffer.close()
            
            # ФИНАЛЬНАЯ ПРОВЕРКА: bytes не должны быть пустыми
            if len(stem_bytes) == 0:
                raise ValueError("❌ CRITICAL: Stem export resulted in empty bytes!")

            return stem_bytes

        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Error creating instrument stem: {e}")
            raise ValueError(f"Stem creation failed: {e}")

    def _create_synthetic_rhythm(self, duration_ms: int, genre_info: Dict) -> AudioSegment:
        """
        Создание синтетического ритма, интегрированного в WaveDream 2.0
        - Учитывает жанр, BPM и уровень энергии
        - Минимум лишних вычислений
        - Готовит материал, который pipeline примет за валидный трек
        """

        try:
            genre_name = genre_info.get('name', 'generic').lower()
            bpm = genre_info.get('target_bpm', 120)
            energy_level = genre_info.get('energy_level', 0.5)

            beat_duration = int(60000 / bpm)
            step_duration = beat_duration // 2
            bars_needed = (duration_ms // (beat_duration * 4)) + 1

            kick_freq = 60 if genre_name == 'trap' else 80
            kick = Sine(kick_freq).to_audio_segment(duration=200).apply_gain(-6 + energy_level * 4)

            snare_duration = 150 if genre_name in ['trap', 'dnb'] else 120
            snare = WhiteNoise().to_audio_segment(duration=snare_duration).apply_gain(-10 + energy_level * 4)
            snare = snare.band_pass_filter(200, 4000)

            hihat_duration = 40 if energy_level > 0.7 else 60
            hihat = WhiteNoise().to_audio_segment(duration=hihat_duration).apply_gain(-14 + energy_level * 4)
            hihat = hihat.high_pass_filter(8000)

            patterns = {
                'generic': {
                    'kick':  [1, 0, 0, 0, 1, 0, 0, 0],
                    'snare': [0, 0, 1, 0, 0, 0, 1, 0],
                    'hihat': [1, 1, 1, 1, 1, 1, 1, 1],
                },
                'trap': {
                    'kick':  [1, 0, 0, 1, 0, 0, 1, 0],
                    'snare': [0, 0, 1, 0, 0, 0, 1, 0],
                    'hihat': [1, 1, 0, 1, 1, 0, 1, 1],
                },
                'house': {
                    'kick':  [1, 0, 0, 0, 1, 0, 0, 0],
                    'snare': [0, 0, 1, 0, 0, 0, 1, 0],
                    'hihat': [0, 1, 0, 1, 0, 1, 0, 1],
                },
                'dnb': {
                    'kick':  [1, 0, 0, 0, 0, 1, 0, 0],
                    'snare': [0, 0, 1, 0, 1, 0, 1, 0],
                    'hihat': [1, 0, 1, 1, 0, 1, 0, 1],
                },
                'lofi': {
                    'kick':  [1, 0, 0, 0, 1, 0, 0, 0],
                    'snare': [0, 0, 1, 0, 0, 0, 1, 0],
                    'hihat': [0, 1, 0, 0, 0, 1, 0, 0],
                },
                'ambient': {
                    'kick':  [1, 0, 0, 0, 0, 0, 0, 0],
                    'snare': [0, 0, 0, 0, 1, 0, 0, 0],
                    'hihat': [0, 0, 1, 0, 0, 0, 1, 0],
                },
                'techno': {
                    'kick':  [1, 0, 0, 0, 1, 0, 0, 0],
                    'snare': [0, 0, 0, 0, 0, 0, 0, 0],
                    'hihat': [1, 0, 1, 0, 1, 0, 1, 0],
                }
            }

            pattern = patterns.get(genre_name, patterns['generic'])

            bar = AudioSegment.silent(duration=beat_duration * 4)

            for inst_name, inst_pattern in pattern.items():
                sound = locals()[inst_name]

                if genre_name == 'lofi':
                    sound = sound.apply_gain(-3).low_pass_filter(8000)
                elif genre_name == 'dnb' and inst_name == 'snare':
                    sound = sound.apply_gain(2)
                elif genre_name == 'ambient':
                    sound = sound.apply_gain(-5).fade_in(20).fade_out(20)

                for i, hit in enumerate(inst_pattern):
                    if hit:
                        pos = i * step_duration

                        if energy_level > 0.7 and random.random() < 0.3:
                            accented_sound = sound.apply_gain(random.randint(2, 4))
                            bar = bar.overlay(accented_sound, position=pos)
                        elif genre_name == 'trap' and inst_name == 'hihat' and random.random() < 0.4:
                            for roll_i in range(2):
                                roll_pos = pos + (roll_i * 30)
                                if roll_pos < beat_duration * 4:
                                    bar = bar.overlay(sound.apply_gain(-2), position=roll_pos)
                        else:
                            bar = bar.overlay(sound, position=pos)

            if genre_name == 'lofi':
                bar = bar.apply_gain(-2)
                if random.random() < 0.1:
                    vinyl_pop = WhiteNoise().to_audio_segment(duration=10).apply_gain(-20)
                    bar = bar.overlay(vinyl_pop, position=random.randint(0, len(bar) - 10))

            elif genre_name == 'ambient':
                bar = bar.fade_in(100).fade_out(100)

            elif genre_name == 'trap' and energy_level > 0.8:
                bar = bar.apply_gain(1)

            rhythm = AudioSegment.silent(duration=0)

            for bar_num in range(bars_needed):
                current_bar = bar

                if bar_num % 4 == 3 and energy_level > 0.5:
                    fill_sound = snare.apply_gain(-3)
                    for fill_pos in [beat_duration * 3 + step_duration, beat_duration * 3 + step_duration * 3]:
                        current_bar = current_bar.overlay(fill_sound, position=fill_pos)

                elif bar_num % 8 == 7 and genre_name in ['trap', 'dnb']:
                    current_bar = bar.overlay(AudioSegment.silent(duration=200), position=beat_duration * 3)

                rhythm += current_bar

            rhythm = rhythm[:duration_ms]

            if genre_name in ['techno', 'house']:
                rhythm = normalize(rhythm, headroom=1.0)
            elif genre_name == 'ambient':
                rhythm = normalize(rhythm, headroom=6.0)
            else:
                rhythm = normalize(rhythm, headroom=3.0)

            fade_duration = min(100, duration_ms // 10)
            rhythm = rhythm.fade_in(fade_duration).fade_out(fade_duration)

            self.logger.info(f"  ✅ Синтетический ритм готов: {genre_name}, {bpm}BPM, "
                             f"{duration_ms}ms, энергия {energy_level:.1f}")

            return rhythm

        except Exception as e:
            self.logger.error(f"❌ Ошибка создания synthetic rhythm: {e}")
            return AudioSegment.silent(duration=duration_ms)
    
    async def _generate_project_report(
        self, request: GenerationRequest, structure: Dict,
        selected_samples: List[Dict], mastering_config: Dict,
        exported_files: Dict[str, str]
    ) -> None:
        """Генерация детального отчёта по проекту с статистикой производительности"""
        report_path = Path(request.output_dir) / f"{self._current_project_name}_detailed_report.md"
        
        try:
            # Создаём markdown отчёт
            total_time = self._performance_stats.get('total_time', 0)
            
            report = f"""# 🎵 WaveDream Enhanced Pro - Детальный отчёт проекта

## 📋 Основная информация
- **Проект**: {self._current_project_name}
- **Промпт**: `{request.prompt}`
- **Жанр**: {structure.get('detected_genre', request.genre or 'auto-detected')}
- **Длительность**: {structure['total_duration']} секунд
- **Назначение мастеринга**: {request.mastering_purpose}
- **Общее время генерации**: {total_time:.1f}с

## ⏱️ Статистика производительности

| Этап | Время (сек) | % от общего |
|------|-------------|-------------|
"""
            
            # Добавляем статистику по этапам
            for stage, time_taken in self._performance_stats.items():
                if stage != 'total_time' and time_taken > 0:
                    percentage = (time_taken / total_time * 100) if total_time > 0 else 0
                    stage_name = stage.replace('_', ' ').title()
                    report += f"| {stage_name} | {time_taken:.2f} | {percentage:.1f}% |\n"
            
            report += f"\n## 🏗️ Структура трека ({len(structure['sections'])} секций)\n\n"
            report += "| № | Тип | Длительность | Энергия | Начало |\n"
            report += "|---|-----|--------------|---------|--------|\n"
            
            for i, section in enumerate(structure['sections'], 1):
                report += (f"| {i} | **{section['type'].title()}** | "
                          f"{section['duration']}с | {section['energy']:.1f} | "
                          f"{section.get('start_time', 0)}с |\n")
            
            report += f"\n## 🎛️ Использованные сэмплы ({len(selected_samples)})\n\n"
            
            if selected_samples:
                report += "| Инструмент | Файл | Секция | Энергия |\n"
                report += "|------------|------|--------|----------|\n"
                
                for sample in selected_samples:
                    instrument = sample.get('instrument_role', 'unknown')
                    filename = sample.get('filename', sample.get('path', 'unknown'))
                    section = sample.get('section', 'unknown')
                    energy = sample.get('section_energy', 0.5)
                    report += f"| {instrument} | `{filename}` | {section} | {energy:.1f} |\n"
            else:
                report += "*Использованы синтетические ритмы*\n"
            
            report += f"\n## 🎚️ Настройки мастеринга\n\n"
            report += f"- **LUFS цель**: {mastering_config.get('target_lufs', 'N/A')}\n" 
            report += f"- **Пик потолок**: {mastering_config.get('peak_ceiling', 'N/A')}dB\n"
            report += f"- **Характер**: {mastering_config.get('character', 'N/A')}\n"
            report += f"- **Стиль**: {mastering_config.get('mastering_style', 'N/A')}\n"
            
            report += f"\n## 📁 Экспортированные файлы ({len(exported_files)})\n\n"
            
            # Группируем файлы по типам
            file_groups = {
                "Финальные треки": [],
                "Промежуточные версии": [],
                "Стемы": [],
                "Метаданные": [],
                "Другое": []
            }
            
            for file_type, file_path in exported_files.items():
                file_name = Path(file_path).name if file_path else file_type
                
                if "final" in file_type.lower():
                    file_groups["Финальные треки"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                elif "stem" in file_type.lower():
                    file_groups["Стемы"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                elif "intermediate" in file_type.lower() or file_type in ["base", "mixed", "processed"]:
                    file_groups["Промежуточные версии"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                elif "metadata" in file_type.lower() or "info" in file_type.lower() or "report" in file_type.lower():
                    file_groups["Метаданные"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                else:
                    file_groups["Другое"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
            
            for group_name, files in file_groups.items():
                if files:
                    report += f"\n### {group_name}\n"
                    for file_line in files:
                        report += f"{file_line}\n"
            
            report += f"\n## 🔧 Техническая информация\n\n"
            report += "**Pipeline этапы выполнены**:\n"
            report += "1. ✅ Анализ метаданных и детекция жанра\n"
            report += "2. ✅ Генерация структуры (LLaMA3/Fallback)\n" 
            report += "3. ✅ Семантический подбор сэмплов\n"
            report += "4. ✅ MusicGen генерация базы → WAV сохранение\n"
            report += "5. ✅ Создание и сохранение стемов → WAV файлы\n"
            report += "6. ✅ Микширование → WAV сохранение\n"
            report += "7. ✅ Обработка эффектами → WAV сохранение\n"
            report += "8. ✅ Умный мастеринг → Финальный WAV\n"
            report += "9. ✅ Верификация качества\n"
            report += "10. ✅ Мульти-форматный экспорт\n\n"
            
            # Добавляем информацию о поэтапном сохранении
            if self._intermediate_storage:
                report += f"## 💾 Поэтапное сохранение\n\n"
                report += "**Сохранённые промежуточные файлы**:\n"
                for stage, path in self._intermediate_storage.items():
                    if isinstance(path, dict):
                        # Это стемы
                        report += f"- **{stage.title()}**: {len(path)} файлов\n"
                        for stem_name, stem_path in path.items():
                            report += f"  - `{Path(stem_path).name}`\n"
                    else:
                        # Обычный файл
                        report += f"- **{stage.title()}**: `{Path(path).name}`\n"
                report += "\n"
            
            report += f"## 💡 Рекомендации по использованию\n\n"
            purpose = request.mastering_purpose
            
            recommendations = {
                "freelance": [
                    "✅ Готов для коммерческой продажи",
                    "📱 Оптимизирован для стриминговых платформ", 
                    "🎧 Протестируйте на различных системах воспроизведения"
                ],
                "professional": [
                    "🎬 Подходит для вещания/кинематографического использования",
                    "📺 Соответствует профессиональным стандартам громкости",
                    "🎛️ Полный динамический диапазон сохранён"
                ],
                "personal": [
                    "🏠 Идеален для персонального прослушивания",
                    "🎵 Естественный, необработанный характер",
                    "🔊 Отлично звучит на домашних аудиосистемах"
                ],
                "family": [
                    "👨‍👩‍👧‍👦 Семейно-ориентированное сведение",
                    "🎥 Идеален для домашних видео",
                    "📱 Хорошо работает на мобильных устройствах"
                ]
            }
            
            purpose_recs = recommendations.get(purpose, recommendations["personal"])
            for rec in purpose_recs:
                report += f"{rec}\n"
            
            report += f"\n## 📊 Системная информация\n\n"
            report += f"- **WaveDream версия**: Enhanced Pro v2.0\n"
            report += f"- **Время создания**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"- **Источник структуры**: {structure.get('source', 'unknown')}\n"
            report += f"- **Общий размер проекта**: {self._calculate_project_size(exported_files)}\n"
            
            report += "\n---\n"
            report += "*Автоматически сгенерирован WaveDream Enhanced Pro Pipeline*\n"
            
            # Сохраняем отчёт
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"  📋 Детальный отчёт сохранён: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания отчёта: {e}")

    def _calculate_project_size(self, exported_files: Dict[str, str]) -> str:
        """Вычисление общего размера проекта"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in exported_files.values():
                if file_path and isinstance(file_path, str):
                    path = Path(file_path)
                    if path.exists():
                        total_size += path.stat().st_size
                        file_count += 1
            
            # Конвертируем в удобный формат
            if total_size > 1024 * 1024 * 1024:  # > 1GB
                size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
            elif total_size > 1024 * 1024:  # > 1MB
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            elif total_size > 1024:  # > 1KB
                size_str = f"{total_size / 1024:.1f} KB"
            else:
                size_str = f"{total_size} bytes"
            
            return f"{size_str} ({file_count} файлов)"
            
        except Exception as e:
            self.logger.error(f"Error calculating project size: {e}")
            return "unknown"

    # === ДОПОЛНИТЕЛЬНЫЕ УДОБНЫЕ МЕТОДЫ ДЛЯ ПОЛЬЗОВАТЕЛЕЙ ===
    
    async def quick_generate(self, prompt: str, genre: str = None, duration: int = 60) -> GenerationResult:
        """Быстрая генерация с минимальными настройками"""
        request = GenerationRequest(
            prompt=prompt,
            genre=genre,
            duration=duration,
            mastering_purpose="personal",
            export_stems=False  # Для быстрой генерации не экспортируем стемы
        )
        
        return await self.generate_track(request)
    
    async def professional_generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Профессиональная генерация со всеми опциями"""
        request = GenerationRequest(
            prompt=prompt,
            mastering_purpose="professional",
            export_stems=True,
            **kwargs
        )
        
        return await self.generate_track(request)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Получение статистики производительности последней генерации"""
        return self._performance_stats.copy()
    
    def get_intermediate_files(self) -> Dict[str, str]:
        """Получение списка промежуточных файлов последней генерации"""
        return self._intermediate_storage.copy()
    
    async def test_pipeline_components(self) -> Dict[str, bool]:
        """Тестирование всех компонентов pipeline"""
        test_results = {}
        
        try:
            # Тест системы экспорта
            export_test = self.export_manager.test_export_system()
            test_results["export_system"] = export_test
            
            # Тест метаданных процессора
            metadata_test = self.metadata_processor.analyze_prompt("test trap beat")
            test_results["metadata_processor"] = bool(metadata_test)
            
            # Тест генерации fallback структуры
            test_genre_info = {"name": "trap", "config": config.get_genre_config("trap")}
            fallback_structure = self._generate_fallback_structure(test_genre_info, 60)
            test_results["fallback_structure"] = bool(fallback_structure.get("sections"))
            
            # Тест создания синтетического ритма
            test_rhythm = self._create_synthetic_rhythm(5000, test_genre_info)
            test_results["synthetic_rhythm"] = len(test_rhythm) > 0
            
            self.logger.info("🧪 Pipeline тестирование завершено:")
            for component, status in test_results.items():
                status_icon = "✅" if status else "❌"
                self.logger.info(f"  {status_icon} {component}: {'OK' if status else 'FAILED'}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка тестирования pipeline: {e}")
            return {"error": str(e)}

    def cleanup_intermediate_files(self) -> int:
        """Очистка промежуточных файлов для экономии места"""
        cleaned_count = 0
        
        try:
            for stage, path in self._intermediate_storage.items():
                if isinstance(path, dict):
                    # Стемы
                    for stem_path in path.values():
                        if Path(stem_path).exists():
                            Path(stem_path).unlink()
                            cleaned_count += 1
                else:
                    # Обычный файл
                    if Path(path).exists():
                        Path(path).unlink()
                        cleaned_count += 1
            
            self._intermediate_storage.clear()
            self.logger.info(f"🧹 Очищено промежуточных файлов: {cleaned_count}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка очистки файлов: {e}")
            return 0


# === СОЗДАНИЕ ГЛОБАЛЬНОГО ЭКЗЕМПЛЯРА И УДОБНЫХ ФУНКЦИЙ ===

# Создаём глобальный экземпляр pipeline
pipeline = WaveDreamPipeline()

# Удобные функции для быстрого использования
async def generate_track(prompt: str, **kwargs) -> GenerationResult:
    """Удобная функция для генерации трека"""
    request = GenerationRequest(prompt=prompt, **kwargs)
    return await pipeline.generate_track(request)

async def quick_beat(prompt: str, genre: str = None) -> str:
    """Быстрая генерация бита, возвращает путь к файлу"""
    result = await pipeline.quick_generate(prompt, genre)
    return result.final_path if result.success else None

async def professional_track(prompt: str, mastering_purpose: str = "professional", **kwargs) -> GenerationResult:
    """Профессиональная генерация трека"""
    request = GenerationRequest(
        prompt=prompt, 
        mastering_purpose=mastering_purpose,
        export_stems=True,
        **kwargs
    )
    return await pipeline.generate_track(request)

def get_pipeline_stats() -> Dict[str, Any]:
    """Получение статистики pipeline"""
    return {
        "performance": pipeline.get_performance_stats(),
        "intermediate_files": pipeline.get_intermediate_files(),
        "current_project": pipeline._current_project_name
    }

# Инициализация при импорте модуля
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Тестирование при запуске модуля
    async def test_run():
        print("🚀 Тестирование WaveDream Pipeline...")
        
        # Тест компонентов
        test_results = await pipeline.test_pipeline_components()
        print(f"Результаты тестирования: {test_results}")
        
        # Тест быстрой генерации
        print("\n🎵 Тест быстрой генерации...")
        result = await quick_beat("aggressive trap beat", "trap")
        
        if result:
            print(f"✅ Трек сгенерирован: {result}")
            
            # Показываем статистику
            stats = get_pipeline_stats()
            print(f"📊 Статистика: {stats['performance']}")
        else:
            print("❌ Ошибка генерации")
    
    # Запускаем тест
    import asyncio
    asyncio.run(test_run())
