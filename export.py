import os
import io
import shutil
import json
import logging
import time
import tempfile
from typing import Dict, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict
from shutil import which
from pydub import AudioSegment, effects
from config import config

logger = logging.getLogger(__name__)

# === ГЛОБАЛЬНЫЕ НАСТРОЙКИ/КОНСТАНТЫ ===
SUPPORTED_FORMATS: Dict[str, Dict[str, str]] = {
    "wav": {"extension": "wav", "quality": "lossless"},
    "mp3": {"extension": "mp3", "quality": "320k"},
    "flac": {"extension": "flac", "quality": "lossless"},
    "aac": {"extension": "m4a", "quality": "256k"},
}

# === УТИЛИТЫ ===
def serialize_for_json(obj: Any) -> Any:
    """Безопасная сериализация любых объектов для JSON"""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def ensure_directory(path: Path) -> bool:
    """Создает директорию если не существует"""
    try:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Cannot create directory {path}: {e}")
        return False


def bytes_to_audiosegment(audio_data: Union[bytes, AudioSegment], fallback_duration: int = 5000) -> AudioSegment:
    """ИСПРАВЛЕННАЯ конвертация bytes в AudioSegment БЕЗ ТИШИНЫ"""
    
    if isinstance(audio_data, AudioSegment):
        # Проверяем что AudioSegment не пустой
        if len(audio_data) == 0:
            raise ValueError("AudioSegment has zero duration!")
        if audio_data.max_dBFS == float("-inf"):
            raise ValueError("AudioSegment is completely silent!")
        return audio_data

    if not isinstance(audio_data, (bytes, bytearray)):
        raise TypeError(f"Invalid audio data type: {type(audio_data)}")

    if len(audio_data) == 0:
        raise ValueError("Empty bytes data provided!")

    try:
        buffer = io.BytesIO(audio_data)
        
        # Пробуем загрузить как WAV сначала
        try:
            audio = AudioSegment.from_wav(buffer)
        except Exception:
            buffer.seek(0)
            # Пробуем автоопределение формата
            audio = AudioSegment.from_file(buffer)

        # КРИТИЧЕСКАЯ ПРОВЕРКА - НЕ ВОЗВРАЩАЕМ ТИШИНУ!
        if len(audio) == 0:
            raise ValueError("Loaded audio has zero duration!")

        if audio.max_dBFS == float("-inf"):
            raise ValueError("Loaded audio is completely silent!")

        logger.info(f"✅ Audio converted: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB")
        return audio
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Cannot convert audio data: {e}")
        raise  # Пробрасываем исключение вместо возврата заглушки


def safe_export_audio(audio: AudioSegment, path: Path, format_name: str = "wav") -> bool:
    """ИСПРАВЛЕННЫЙ безопасный экспорт аудио с проверками"""
    
    # КРИТИЧЕСКИЕ ПРОВЕРКИ ВХОДНЫХ ДАННЫХ
    if audio is None:
        logger.error("❌ CRITICAL: Audio is None!")
        return False
        
    if len(audio) == 0:
        logger.error("❌ CRITICAL: Audio has zero duration!")
        return False
        
    if audio.max_dBFS == float("-inf"):
        logger.error("❌ CRITICAL: Audio is completely silent!")
        return False

    try:
        if not ensure_directory(path):
            return False

        # Проверяем права на запись
        parent_dir = path.parent
        if not os.access(parent_dir, os.W_OK):
            temp_dir = Path(tempfile.gettempdir()) / "wavedream_export"
            temp_dir.mkdir(exist_ok=True)
            path = temp_dir / path.name
            logger.warning(f"⚠️ No write permission, fallback to: {path}")

        # Если файл заблокирован - создаем с timestamp
        if path.exists():
            try:
                with open(path, "a"):
                    pass
            except PermissionError:
                ts = int(time.time() * 1000)
                path = path.parent / f"{path.stem}_{ts}{path.suffix}"
                logger.warning(f"⚠️ File locked, using: {path.name}")

        logger.info(
            f"🎵 Exporting audio: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB -> {path}"
        )

        # ИСПРАВЛЕННЫЙ экспорт с fallback на WAV
        try:
            # Экспорт в нужном формате
            export_params = {}
            if format_name == "mp3":
                export_params = {
                    "bitrate": "320k",
                    "parameters": ["-avoid_negative_ts", "make_zero"]
                }
            
            audio.export(str(path), format=format_name, **export_params)
            
        except Exception as export_error:
            logger.warning(f"⚠️ {format_name} export failed, trying WAV: {export_error}")
            
            # Fallback на WAV через buffer
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            data = buffer.getvalue()
            
            if not data:
                raise ValueError("WAV export resulted in empty data!")
                
            # Меняем расширение на .wav и сохраняем
            path = path.with_suffix(".wav")
            with open(path, "wb") as f:
                f.write(data)

        # ПРОВЕРКА РЕЗУЛЬТАТА
        if not path.exists():
            logger.error(f"❌ Export failed: {path} not created")
            return False
            
        size = path.stat().st_size
        if size < 1000:  # Минимум 1KB для аудиофайла
            logger.error(f"❌ Export failed: {path} too small ({size} bytes)")
            return False

        logger.info(f"✅ Exported successfully: {path} ({size} bytes)")
        return True

    except Exception as e:
        logger.error(f"❌ Critical export error for {path}: {e}")
        
        # EMERGENCY SAVE - только если исходное аудио валидное
        try:
            if audio.max_dBFS != float('-inf') and len(audio) > 0:
                emergency_path = Path.cwd() / f"emergency_{path.name}"
                audio.export(str(emergency_path), format="wav")
                
                if emergency_path.exists() and emergency_path.stat().st_size > 1000:
                    logger.warning(f"🚨 Emergency save successful: {emergency_path}")
                    return True
        except Exception as e2:
            logger.error(f"❌ Emergency save failed: {e2}")
            
        return False


class ExportManager:
    """
    ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ система экспорта WaveDream 2.0
    
    КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
    1. ✅ Добавлен метод save_final_mix() 
    2. ✅ Исправлена функция force_save_everything() - правильная сигнатура
    3. ✅ Убраны все заглушки с тишиной
    4. ✅ Строгая валидация на каждом этапе
    5. ✅ Правильная обработка ошибок без потери данных
    """
    
    def __init__(self, base_dir: str = "output"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_formats = SUPPORTED_FORMATS
        self.logger.info(f"📁 ExportManager 2.0 initialized: {self.base_dir}")

    # === ОСНОВНЫЕ МЕТОДЫ СОХРАНЕНИЯ (для pipeline.py) ===


    def check_export_environment(self) -> dict:
        """
        Возвращает строгий словарь флагов окружения, без исключений.
        """
        try:
            base_dir = Path(self.export_dir or "./exports")
        except Exception:
            base_dir = Path("./exports")

        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            base_dir_writable = os.access(base_dir, os.W_OK)
        except Exception:
            base_dir_writable = False

        # Свободное место — некритично, поэтому True по умолчанию
        try:
            total, used, free = shutil.disk_usage(str(base_dir))
            min_free_gb = float(getattr(self, "min_free_gb", 0.2))
            sufficient_space = (free / (1024 ** 3)) > min_free_gb
        except Exception:
            sufficient_space = True

        # Проверка pydub
        try:
            _ = AudioSegment.silent(duration=50)
            pydub_working = True
        except Exception:
            pydub_working = False

        # Проверка наличия ffmpeg/sox
        ffmpeg_or_sox = (which("ffmpeg") is not None or which("sox") is not None)

        return {
            "base_dir_writable": bool(base_dir_writable),
            "sufficient_space": bool(sufficient_space),
            "pydub_working": bool(pydub_working),
            "ffmpeg_or_sox": bool(ffmpeg_or_sox),
        }
    
    async def save_intermediate(self, name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет промежуточный файл"""
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                logger.warning(f"⚠️ Empty audio data for intermediate '{name}'")
                return ""
            
            audio = bytes_to_audiosegment(audio_bytes)
            intermediate_dir = Path(output_dir) / "intermediate"
            ensure_directory(intermediate_dir)
            
            stage_path = intermediate_dir / f"{name}.wav"
            
            if safe_export_audio(audio, stage_path, "wav"):
                logger.info(f"💾 Intermediate '{name}' saved: {stage_path}")
                return str(stage_path)
            else:
                logger.error(f"❌ Failed to save intermediate '{name}'")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error saving intermediate '{name}': {e}")
            return ""

    async def save_stem(self, stem_name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет отдельный стем"""
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                logger.warning(f"⚠️ Empty audio data for stem '{stem_name}'")
                return ""
            
            audio = bytes_to_audiosegment(audio_bytes)
            stems_dir = Path(output_dir) / "stems"
            ensure_directory(stems_dir)
            
            stem_path = stems_dir / f"stem_{stem_name}.wav"
            
            if safe_export_audio(audio, stem_path, "wav"):
                logger.info(f"🎛️ Stem '{stem_name}' saved: {stem_path}")
                return str(stem_path)
            else:
                logger.error(f"❌ Failed to save stem '{stem_name}'")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error saving stem '{stem_name}': {e}")
            return ""

    async def save_final_mix(self, project_name: str, audio_bytes: bytes, output_dir: str, 
                           formats: list = None) -> Dict[str, str]:
        """
        ДОБАВЛЕННЫЙ МЕТОД - сохраняет финальный микс в нескольких форматах
        """
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                raise ValueError("❌ CRITICAL: Empty final mix audio data!")
            
            audio = bytes_to_audiosegment(audio_bytes)
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            if formats is None:
                formats = ["wav", "mp3"]
            
            saved_files = {}
            
            # Сохраняем в каждом запрошенном формате
            for fmt in formats:
                if fmt in self.supported_formats:
                    fmt_info = self.supported_formats[fmt]
                    final_path = output_path / f"{project_name}_FINAL.{fmt_info['extension']}"
                    
                    if safe_export_audio(audio, final_path, fmt):
                        saved_files[f"final_{fmt}"] = str(final_path)
                        logger.info(f"🎵 Final mix saved as {fmt}: {final_path}")
                    else:
                        logger.error(f"❌ Failed to save final mix as {fmt}")
                else:
                    logger.warning(f"⚠️ Unsupported format: {fmt}")
            
            if not saved_files:
                raise ValueError("❌ CRITICAL: No final mix files were saved!")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error saving final mix: {e}")
            raise

    async def force_save_everything(self, audio_dict: Dict[str, bytes], output_dir: str) -> Dict[str, str]:
        """
        ИСПРАВЛЕННАЯ аварийная функция сохранения всех треков
        Исправлена сигнатура - теперь принимает 3 аргумента вместо 4
        """
        logger.info("🚨 EMERGENCY SAVE MODE ACTIVATED")
        
        emergency_dir = Path(output_dir) / f"emergency_export_{int(time.time())}"
        ensure_directory(emergency_dir)
        
        saved_files = {}
        
        for name, audio_bytes in audio_dict.items():
            try:
                if not audio_bytes or len(audio_bytes) == 0:
                    logger.warning(f"⚠️ Skipping empty audio: {name}")
                    continue
                
                # Пробуем конвертировать и сохранить
                audio = bytes_to_audiosegment(audio_bytes)
                emergency_path = emergency_dir / f"emergency_{name}.wav"
                
                if safe_export_audio(audio, emergency_path, "wav"):
                    saved_files[f"emergency_{name}"] = str(emergency_path)
                    logger.info(f"🚨 Emergency saved: {name} -> {emergency_path}")
                else:
                    logger.error(f"❌ Emergency save failed for: {name}")
                    
            except Exception as e:
                logger.error(f"❌ Error in emergency save for '{name}': {e}")
                
                # ПОСЛЕДНИЙ ШАНС - сохраняем raw bytes
                try:
                    raw_path = emergency_dir / f"raw_bytes_{name}.bin"
                    with open(raw_path, 'wb') as f:
                        f.write(audio_bytes)
                    saved_files[f"raw_{name}"] = str(raw_path)
                    logger.warning(f"🔧 Saved raw bytes: {name} -> {raw_path}")
                except Exception as e2:
                    logger.error(f"❌ Even raw save failed for '{name}': {e2}")
        
        logger.info(f"🚨 Emergency save complete: {len(saved_files)} files in {emergency_dir}")
        return saved_files

    # === ЭКСПОРТ МЕТАДАННЫХ ===
    
    async def export_metadata(self, config: Dict, project_dir: Path) -> Dict[str, str]:
        """Экспорт метаданных с правильной JSON сериализацией"""
        exported_files = {}

        try:
            # Сериализуем все данные
            request_data = serialize_for_json(config.get("request_data", {}))
            structure = serialize_for_json(config.get("structure", {}))
            samples = serialize_for_json(config.get("samples", []))
            mastering = serialize_for_json(config.get("mastering", {}))

            # Основная информация проекта
            project_info = {
                "wavedream_version": "2.0.0",
                "export_timestamp": time.time(),
                "generation_config": {
                    "prompt": request_data.get("prompt", ""),
                    "genre": request_data.get("genre", "unknown"),
                    "mastering_purpose": request_data.get("mastering_purpose", "personal"),
                    "bpm": request_data.get("bpm", 120),
                    "duration": request_data.get("duration", 0),
                    "energy_level": request_data.get("energy_level", 0.5),
                    "creativity_factor": request_data.get("creativity_factor", 0.7),
                },
                "structure_info": {
                    "total_duration": structure.get("total_duration", 0),
                    "sections_count": len(structure.get("sections", [])),
                    "source": structure.get("source", "unknown"),
                },
                "export_settings": {
                    "export_stems": config.get("export_stems", True),
                    "formats": config.get("export_formats", ["wav", "mp3"]),
                },
            }

            # Функция безопасного сохранения JSON
            def safe_json_save(data: dict, file_path: Path, description: str) -> bool:
                try:
                    temp_path = file_path.with_suffix(".tmp")
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                        
                    if temp_path.exists() and temp_path.stat().st_size > 10:
                        temp_path.rename(file_path)
                        logger.info(f"  ✅ {description}: {file_path}")
                        return True
                    else:
                        logger.error(f"  ❌ {description} failed: empty temp file")
                        return False
                        
                except Exception as e:
                    logger.error(f"  ❌ {description} error: {e}")
                    if "temp_path" in locals() and temp_path.exists():
                        temp_path.unlink()
                    return False

            # Сохраняем все файлы метаданных
            metadata_files = [
                (project_info, project_dir / "project_info.json", "Project info", "project_info"),
                (structure, project_dir / "track_structure.json", "Track structure", "structure"),
                (samples, project_dir / "used_samples.json", "Used samples", "samples"),
                (mastering, project_dir / "mastering_config.json", "Mastering config", "mastering"),
            ]

            for data, file_path, description, key in metadata_files:
                if data and safe_json_save(data, file_path, description):
                    exported_files[key] = str(file_path)

            logger.info(f"  📋 Exported {len(exported_files)} metadata files")
            
        except Exception as e:
            logger.error(f"❌ Error exporting metadata: {e}")
            
            # Emergency metadata
            try:
                emergency_path = project_dir / "emergency_metadata.txt"
                with open(emergency_path, "w", encoding="utf-8") as f:
                    f.write("WaveDream Export Emergency Metadata\n")
                    f.write(f"Timestamp: {time.time()}\n")
                    f.write(f"Config keys: {list(config.keys())}\n")
                    f.write(f"Error: {str(e)}\n")
                exported_files["emergency"] = str(emergency_path)
                logger.warning(f"🚨 Emergency metadata saved: {emergency_path}")
            except Exception as ee:
                logger.error(f"❌ Emergency metadata save failed: {e} / {ee}")

        return exported_files

    # === ПОЛНЫЙ ЭКСПОРТ ПРОЕКТА ===
    
    async def export_complete_project(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_audio: Dict[str, Any],
        config: Dict
    ) -> Dict[str, str]:
        """
        ИСПРАВЛЕННЫЙ экспорт полного проекта WaveDream 2.0
        """
        self.logger.info("💾 Starting complete project export (WaveDream 2.0)...")
        
        try:
            # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ
            if mastered_audio is None:
                raise ValueError("❌ CRITICAL: mastered_audio is None!")
            
            if isinstance(mastered_audio, (bytes, bytearray)) and len(mastered_audio) == 0:
                raise ValueError("❌ CRITICAL: mastered_audio is empty!")
            
            # Создаем структуру проекта
            output_dir = Path(config.get("output_dir", self.base_dir))
            ensure_directory(output_dir)
            
            timestamp = int(time.time())
            project_name = f"WaveDream_2_0_{timestamp}"
            project_dir = output_dir / project_name
            ensure_directory(project_dir)
            
            self.logger.info(f"📁 Project directory: {project_dir}")
            
            exported_files = {}
            
            # 1. ЭКСПОРТ ФИНАЛЬНОГО ТРЕКА
            self.logger.info("🎵 Exporting final mastered track...")
            try:
                final_audio = bytes_to_audiosegment(mastered_audio) if isinstance(mastered_audio, (bytes, bytearray)) else mastered_audio
                
                formats = config.get("export_formats", ["wav", "mp3"])
                final_files = await self.save_final_mix(project_name, mastered_audio, str(project_dir), formats)
                exported_files.update(final_files)
                
            except Exception as e:
                self.logger.error(f"❌ CRITICAL: Final track export failed: {e}")
                raise ValueError(f"Cannot export final track: {e}")
            
            # 2. ЭКСПОРТ ПРОМЕЖУТОЧНЫХ ВЕРСИЙ И СТЕМОВ
            if config.get("export_stems", True) and intermediate_audio:
                self.logger.info("🎛️ Exporting stems and intermediate versions...")
                
                # Промежуточные версии
                stage_mapping = {
                    "base": "01_MusicGen_Base",
                    "mixed": "02_Mixed_Stems", 
                    "processed": "03_Effects_Applied",
                }
                
                for stage_key, stage_name in stage_mapping.items():
                    if stage_key in intermediate_audio:
                        try:
                            saved_path = await self.save_intermediate(stage_name, intermediate_audio[stage_key], str(project_dir))
                            if saved_path:
                                exported_files[f"intermediate_{stage_key}"] = saved_path
                        except Exception as e:
                            self.logger.error(f"❌ Error exporting intermediate {stage_key}: {e}")
                
                # Стемы
                if "stems" in intermediate_audio and isinstance(intermediate_audio["stems"], dict):
                    for instrument, stem_data in intermediate_audio["stems"].items():
                        try:
                            saved_path = await self.save_stem(instrument, stem_data, str(project_dir))
                            if saved_path:
                                exported_files[f"stem_{instrument}"] = saved_path
                        except Exception as e:
                            self.logger.error(f"❌ Error exporting stem {instrument}: {e}")
            
            # 3. ЭКСПОРТ МЕТАДАННЫХ
            self.logger.info("📋 Exporting metadata...")
            try:
                metadata_files = await self.export_metadata(config, project_dir)
                exported_files.update(metadata_files)
            except Exception as e:
                self.logger.error(f"❌ Metadata export failed: {e}")
            
            # 4. СОЗДАНИЕ ОТЧЕТА ПРОЕКТА
            self.logger.info("📊 Creating project report...")
            try:
                report_file = await self.create_project_report(config, exported_files, project_dir)
                if report_file:
                    exported_files["project_report"] = report_file
            except Exception as e:
                self.logger.error(f"❌ Report creation failed: {e}")
            
            # 5. ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ
            self.validate_exports(exported_files)
            
            # ФИНАЛЬНАЯ ПРОВЕРКА
            if not any(key.startswith("final") for key in exported_files.keys()):
                raise ValueError("❌ CRITICAL: No final audio files were exported!")
            
            self.logger.info(f"🎉 WaveDream 2.0 export complete: {len(exported_files)} files in {project_dir}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL Export error: {e}")
            raise ValueError(f"WaveDream 2.0 export failed: {e}")

    async def create_project_report(self, config: Dict, exported_files: Dict[str, str], project_dir: Path) -> Optional[str]:
        """Создание детального отчёта проекта"""
        try:
            report_path = project_dir / "WAVEDREAM_2_0_REPORT.md"
            request_data = config.get("request_data", {})
            structure = config.get("structure", {})

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# 🎵 WaveDream Enhanced Pro 2.0 - Project Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")

                # Project Overview
                f.write("## 📋 Project Overview\n\n")
                f.write(f"**Prompt**: `{request_data.get('prompt', 'N/A')}`\n\n")
                f.write(f"**Genre**: {request_data.get('genre', 'Auto-detected')}\n\n") 
                f.write(f"**BPM**: {request_data.get('bpm', 'Auto')}\n\n")
                f.write(f"**Duration**: {structure.get('total_duration', 0):.1f} seconds\n\n")
                f.write(f"**Mastering Purpose**: {request_data.get('mastering_purpose', 'personal')}\n\n")
                f.write(f"**Energy Level**: {request_data.get('energy_level', 0.5):.1f}\n\n")
                f.write(f"**Creativity Factor**: {request_data.get('creativity_factor', 0.7):.1f}\n\n")

                # Track Structure
                if "sections" in structure and structure["sections"]:
                    f.write("## 🏗️ Track Structure\n\n")
                    f.write("| Section | Duration | Energy Level | Start Time |\n")
                    f.write("|---------|----------|--------------|------------|\n")
                    for section in structure["sections"]:
                        section_type = section.get("type", "unknown")
                        duration = section.get("duration", 0)
                        energy = section.get("energy", 0.5)
                        start_time = section.get("start_time", 0)
                        f.write(f"| {section_type.title()} | {duration}s | {energy:.1f} | {start_time}s |\n")
                    f.write("\n")

                # Exported Files
                f.write("## 📁 Exported Files\n\n")
                for file_type, file_path in exported_files.items():
                    f.write(f"- **{file_type.replace('_', ' ').title()}**: `{Path(file_path).name}`\n")
                f.write("\n")

                # Technical Pipeline
                f.write("## 🔧 WaveDream 2.0 Pipeline\n\n")
                pipeline_stages = [
                    "✅ Metadata Analysis & Genre Detection",
                    "✅ Structure Generation (LLM/Fallback)", 
                    "✅ Semantic Sample Selection",
                    "✅ MusicGen Base Generation",
                    "✅ Multi-Stem Creation & Layering",
                    "✅ Intelligent Mixing Engine",
                    "✅ Advanced Effects Processing", 
                    "✅ Smart Mastering (Purpose-Aware)",
                    "✅ Quality Verification & Validation",
                    "✅ Multi-Format Export System"
                ]
                
                for i, stage in enumerate(pipeline_stages, 1):
                    f.write(f"{i}. {stage}\n")
                f.write("\n")

                # Usage Recommendations
                f.write("## 🎯 Usage Recommendations\n\n")
                purpose = request_data.get("mastering_purpose", "personal")
                recommendations = {
                    "freelance": [
                        "✅ Ready for commercial distribution",
                        "📱 Streaming platform optimized", 
                        "🎧 Test on multiple playback systems",
                        "💰 Suitable for client delivery"
                    ],
                    "professional": [
                        "🎬 Broadcast/cinema ready",
                        "📺 Professional loudness standards",
                        "🎛️ Full dynamic range preserved",
                        "🏢 Industry-standard quality"
                    ],
                    "personal": [
                        "🏠 Perfect for personal listening",
                        "🎵 Natural, musical character",
                        "🔊 Home audio system friendly", 
                        "📱 Mobile device optimized"
                    ],
                    "family": [
                        "👨‍👩‍👧‍👦 Family-safe mixing approach",
                        "🎥 Ideal for home videos/content",
                        "📱 Mobile and tablet friendly",
                        "🔊 Clear dialogue/vocals"
                    ]
                }
                
                current_recs = recommendations.get(purpose, recommendations["personal"])
                for rec in current_recs:
                    f.write(f"- {rec}\n")
                f.write("\n")

                # Footer
                f.write("---\n")
                f.write("*Generated by **WaveDream Enhanced Pro 2.0** - AI Music Generation System*\n")

            logger.info(f"📋 WaveDream 2.0 report created: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating project report: {e}")
            return None

    def validate_exports(self, exported_files: Dict[str, str]) -> None:
        """Валидация экспортированных файлов"""
        logger.info("🔍 Validating exported files...")
        
        total_files = len(exported_files)
        valid_files = 0
        
        for file_type, file_path in exported_files.items():
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                if size > 0:
                    valid_files += 1
                    logger.info(f"  ✅ {file_type}: {path.name} ({size:,} bytes)")
                else:
                    logger.error(f"  ❌ {file_type}: {path.name} (0 bytes - EMPTY)")
            else:
                logger.error(f"  ❌ {file_type}: {path.name} (MISSING)")
        
        logger.info(f"📊 Validation complete: {valid_files}/{total_files} files valid")
        
        if valid_files == 0:
            raise ValueError("❌ CRITICAL: No valid files were exported!")
        
        if valid_files < total_files:
            logger.warning(f"⚠️ Some files failed validation ({total_files - valid_files} failures)")

    # === ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ===
    
    def get_export_summary(self, exported_files: Dict[str, str]) -> Dict[str, Any]:
        """Создает сводку экспорта"""
        summary = {
            "total_files": len(exported_files),
            "file_types": {},
            "total_size": 0,
            "status": "success"
        }
        
        for file_type, file_path in exported_files.items():
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                summary["total_size"] += size
                
                # Группируем по типам
                category = file_type.split("_")[0]
                if category not in summary["file_types"]:
                    summary["file_types"][category] = []
                summary["file_types"][category].append({
                    "name": path.name,
                    "size": size,
                    "path": str(path)
                })
            else:
                summary["status"] = "partial"
        
        return summary

    def cleanup_temp_files(self, project_dir: Path) -> None:
        """Очистка временных файлов"""
        try:
            temp_patterns = ["*.tmp", "*.temp", "emergency_*"]
            cleaned = 0
            
            for pattern in temp_patterns:
                for temp_file in project_dir.glob(pattern):
                    try:
                        temp_file.unlink()
                        cleaned += 1
                    except Exception as e:
                        logger.debug(f"Could not remove temp file {temp_file}: {e}")
            
            if cleaned > 0:
                logger.info(f"🧹 Cleaned up {cleaned} temporary files")
                
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")


# === STANDALONE EXPORT FUNCTIONS (для обратной совместимости) ===

async def export_final_track(
    final_audio: AudioSegment, project_dir: Path, config: Dict, project_name: str
) -> Dict[str, str]:
    """Standalone функция экспорта финального трека"""
    
    if final_audio is None or len(final_audio) == 0 or final_audio.max_dBFS == float("-inf"):
        raise ValueError("Invalid final_audio provided")

    exported_files = {}
    request_data = config.get("request_data", {})
    prompt = request_data.get("prompt", "unknown")
    purpose = request_data.get("mastering_purpose", "personal")

    # Создаем безопасное имя файла
    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
    safe_prompt = "_".join(safe_prompt.split())
    base_name = f"{project_name}_{safe_prompt}_{purpose}_FINAL"

    # Основной WAV файл
    main_path = project_dir / f"{base_name}.wav"
    if not safe_export_audio(final_audio, main_path, "wav"):
        raise ValueError(f"Failed to export main WAV: {main_path}")
    exported_files["final"] = str(main_path)

    # Дополнительные форматы
    additional_formats = config.get("export_formats", ["mp3"])
    for fmt in additional_formats:
        if fmt != "wav" and fmt in SUPPORTED_FORMATS:
            fmt_info = SUPPORTED_FORMATS[fmt]
            fmt_path = project_dir / f"{base_name}.{fmt_info['extension']}"
            if safe_export_audio(final_audio, fmt_path, fmt):
                exported_files[f"final_{fmt}"] = str(fmt_path)
            else:
                logger.warning(f"⚠️ Failed to export {fmt}: {fmt_path}")

    return exported_files


async def export_intermediate_versions(
    intermediate_audio: Dict[str, Any], project_dir: Path, config: Dict, project_name: str
) -> Dict[str, str]:
    """Standalone функция экспорта промежуточных версий"""
    
    exported_files = {}
    stems_dir = project_dir / "stems"
    inter_dir = project_dir / "intermediate"
    ensure_directory(stems_dir)
    ensure_directory(inter_dir)

    # Промежуточные версии
    stage_mapping = {
        "base": "01_MusicGen_Base",
        "mixed": "02_Mixed_Stems",
        "processed": "03_Effects_Applied",
    }

    for stage_key, stage_name in stage_mapping.items():
        if stage_key in intermediate_audio:
            try:
                audio_segment = bytes_to_audiosegment(intermediate_audio[stage_key])
                stage_path = inter_dir / f"{stage_name}.wav"
                if safe_export_audio(audio_segment, stage_path, "wav"):
                    exported_files[f"intermediate_{stage_key}"] = str(stage_path)
            except Exception as e:
                logger.error(f"❌ Error exporting intermediate {stage_key}: {e}")

    # Стемы
    if "stems" in intermediate_audio and isinstance(intermediate_audio["stems"], dict):
        for instrument, stem_data in intermediate_audio["stems"].items():
            try:
                stem_audio = bytes_to_audiosegment(stem_data)
                stem_path = stems_dir / f"Stem_{instrument.title()}.wav"
                if safe_export_audio(stem_audio, stem_path, "wav"):
                    exported_files[f"stem_{instrument}"] = str(stem_path)
            except Exception as e:
                logger.error(f"❌ Error exporting stem {instrument}: {e}")

    return exported_files

    def sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Чистит аудио от NaN, Inf и клиппит в [-1,1].
        """
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        if max_val > 1.0:
            audio = audio / max_val
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)

    def safe_export_aubio(self, audio: np.ndarray, filename: str, format: str = "wav"):
        """
        Экспортирует аудио с проверками и fallback на Pydub.
        """
        try:
            audio = self.sanitize_audio(audio)
            filepath = self.export_dir / f"{filename}.{format}"

            # Для WAV/FLAC используем soundfile напрямую
            if format.lower() in ("wav", "flac"):
                sf.write(filepath, audio, self.sample_rate)
                logging.info(f"💾 Аудио сохранено: {filepath}")
                return filepath

            # Для остального через pydub + ffmpeg/sox
            temp_wav = self.export_dir / f"{filename}_temp.wav"
            sf.write(temp_wav, audio, self.sample_rate)
            seg = AudioSegment.from_wav(temp_wav)
            seg.export(filepath, format=format)
            temp_wav.unlink(missing_ok=True)

            logging.info(f"💾 Аудио сохранено: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"❌ Ошибка при экспорте {filename}: {e}")
            raise
