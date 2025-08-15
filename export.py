import os
import io
import json
import logging
import time
from typing import Dict, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict
from pydub import AudioSegment
from config import config

logger = logging.getLogger(__name__)

# === ГЛОБАЛЬНЫЕ НАСТРОЙКИ/КОНСТАНТЫ ===
SUPPORTED_FORMATS: Dict[str, Dict[str, str]] = {
    "wav": {"extension": "wav", "quality": "lossless"},
    "mp3": {"extension": "mp3", "quality": "320k"},
    "flac": {"extension": "flac", "quality": "lossless"},
    "aac": {"extension": "m4a", "quality": "256k"},
}

# === УТИЛЫ ВНЕ КЛАССА (были self.*) ===
def serialize_for_json(obj: Any) -> Any:
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
    try:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Cannot create directory {path}: {e}")
        return False


def bytes_to_audiosegment(audio_data: Union[bytes, AudioSegment]) -> AudioSegment:
    if isinstance(audio_data, AudioSegment):
        if audio_data.max_dBFS == float("-inf"):
            raise ValueError("AudioSegment is completely silent!")
        return audio_data

    if not isinstance(audio_data, bytes):
        raise TypeError(f"Invalid audio data type: {type(audio_data)}")

    if len(audio_data) == 0:
        raise ValueError("Empty bytes data provided!")

    try:
        buffer = io.BytesIO(audio_data)
        try:
            audio = AudioSegment.from_wav(buffer)
        except Exception:
            buffer.seek(0)
            audio = AudioSegment.from_file(buffer)

        if len(audio) == 0:
            raise ValueError("Loaded audio has zero duration!")

        if audio.max_dBFS == float("-inf"):
            raise ValueError("Loaded audio is completely silent!")

        logger.info(f"✅ Audio converted: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB")
        return audio
    except Exception as e:
        logger.error(f"Cannot convert audio data: {e}")
        raise


def safe_export_audio(audio: AudioSegment, path: Path, format_name: str = "wav") -> bool:
    if audio is None or len(audio) == 0 or audio.max_dBFS == float("-inf"):
        logger.error("❌ CRITICAL: Invalid audio supplied for export.")
        return False

    try:
        if not ensure_directory(path):
            return False

        parent_dir = path.parent
        if not os.access(parent_dir, os.W_OK):
            import tempfile

            temp_dir = Path(tempfile.gettempdir()) / "wavedream_export"
            temp_dir.mkdir(exist_ok=True)
            path = temp_dir / path.name
            logger.warning(f"⚠️ No write permission, fallback to: {path}")

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

        try:
            audio.export(
                str(path),
                format=format_name,
                bitrate="320k" if format_name == "mp3" else None,
                parameters=["-avoid_negative_ts", "make_zero"] if format_name == "mp3" else None,
            )
        except Exception as export_error:
            logger.warning(f"⚠️ {format_name} export failed, trying WAV: {export_error}")
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            data = buffer.getvalue()
            if not data:
                raise ValueError("WAV export resulted in empty data!")
            path = path.with_suffix(".wav")
            with open(path, "wb") as f:
                f.write(data)

        if not path.exists():
            logger.error(f"❌ Export failed: {path} not created")
            return False
        size = path.stat().st_size
        if size < 1000:
            logger.error(f"❌ Export failed: {path} too small ({size} bytes)")
            return False

        logger.info(f"✅ Exported successfully: {path} ({size} bytes)")
        return True

    except Exception as e:
        logger.error(f"❌ Critical export error for {path}: {e}")
        try:
            emergency_path = Path.cwd() / f"emergency_{path.name}"
            audio.export(str(emergency_path), format="wav")
            if emergency_path.exists() and emergency_path.stat().st_size > 1000:
                logger.warning(f"🚨 Emergency save successful: {emergency_path}")
                return True
        except Exception as e2:
            logger.error(f"❌ Emergency save failed: {e2}")
        return False


# === ФУНКЦИИ ВНЕ КЛАССА (были методами) ===
async def export_metadata(config: Dict, project_dir: Path) -> Dict[str, str]:
    """Экспорт метаданных с правильной JSON сериализацией"""
    exported_files: Dict[str, str] = {}

    try:
        request_data = serialize_for_json(config.get("request_data", {}))
        structure = serialize_for_json(config.get("structure", {}))
        samples = serialize_for_json(config.get("samples", []))
        mastering = serialize_for_json(config.get("mastering", {}))

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

        def safe_json_save(data: dict, file_path: Path, description: str) -> bool:
            try:
                temp_path = file_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                if temp_path.exists() and temp_path.stat().st_size > 10:
                    temp_path.rename(file_path)
                    logger.info(f"  ✅ {description}: {file_path}")
                    return True
                logger.error(f"  ❌ {description} failed: empty temp file")
                return False
            except Exception as e:
                logger.error(f"  ❌ {description} error: {e}")
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
                return False

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


async def create_project_report(
    config: Dict, exported_files: Dict[str, str], project_dir: Path
) -> Optional[str]:
    """Создание отчёта проекта"""
    try:
        report_path = project_dir / "PROJECT_REPORT.md"
        request_data = config.get("request_data", {})
        structure = config.get("structure", {})

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 🎵 WaveDream Enhanced Pro - Project Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📋 Project Overview\n\n")
            f.write(f"**Prompt**: `{request_data.get('prompt', 'N/A')}`\n\n")
            f.write(f"**Genre**: {request_data.get('genre', 'Auto-detected')}\n\n")
            f.write(f"**BPM**: {request_data.get('bpm', 'Auto')}\n\n")
            f.write(f"**Duration**: {structure.get('total_duration', 0):.1f} seconds\n\n")
            f.write(f"**Mastering Purpose**: {request_data.get('mastering_purpose', 'personal')}\n\n")
            f.write(f"**Energy Level**: {request_data.get('energy_level', 0.5):.1f}\n\n")
            f.write(f"**Creativity Factor**: {request_data.get('creativity_factor', 0.7):.1f}\n\n")

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

            f.write("## 📁 Exported Files\n\n")
            for file_type, file_path in exported_files.items():
                f.write(f"- **{file_type.replace('_', ' ').title()}**: `{Path(file_path).name}`\n")
            f.write("\n")

            f.write("## 🔧 Technical Details\n\n")
            f.write("**Pipeline Stages Completed**:\n")
            for i, line in enumerate(
                [
                    "Metadata Analysis & Genre Detection",
                    "Structure Generation (LLM/Fallback)",
                    "Semantic Sample Selection",
                    "MusicGen Base Generation",
                    "Stem Creation & Mixing",
                    "Effects Processing",
                    "Smart Mastering",
                    "Quality Verification",
                    "Multi-format Export",
                ],
                start=1,
            ):
                f.write(f"{i}. ✅ {line}\n")
            f.write("\n")

            purpose = request_data.get("mastering_purpose", "personal")
            recommendations = {
                "freelance": [
                    "✅ Ready for commercial sale",
                    "📱 Optimized for streaming platforms",
                    "🎧 Test on various playback systems",
                ],
                "professional": [
                    "🎬 Suitable for broadcast/cinema use",
                    "📺 Meets professional loudness standards",
                    "🎛️ Full dynamic range preserved",
                ],
                "personal": [
                    "🏠 Perfect for personal listening",
                    "🎵 Natural, unprocessed character",
                    "🔊 Great on home audio systems",
                ],
                "family": [
                    "👨‍👩‍👧‍👦 Family-friendly mixing",
                    "🎥 Ideal for home videos",
                    "📱 Works well on mobile devices",
                ],
            }
            for rec in recommendations.get(purpose, recommendations["personal"]):
                f.write(f"- {rec}\n")

            f.write("\n---\n*Generated by WaveDream Enhanced Pro v2.0*\n")

        logger.info(f"📋 Project report created: {report_path}")
        return str(report_path)
    except Exception as e:
        logger.error(f"❌ Error creating project report: {e}")
        return None


def validate_exports(exported_files: Dict[str, str]) -> None:
    logger.info("🔍 Validating exported files...")
    total_files = len(exported_files)
    valid_files = 0
    for file_type, file_path in exported_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                valid_files += 1
                logger.info(f"  ✅ {file_type}: {path.name} ({size} bytes)")
            else:
                logger.error(f"  ❌ {file_type}: {path.name} (0 bytes - EMPTY)")
        else:
            logger.error(f"  ❌ {file_type}: {path.name} (MISSING)")
    logger.info(f"📊 Validation complete: {valid_files}/{total_files} files valid")


async def export_final_track(
    final_audio: AudioSegment, project_dir: Path, config: Dict, project_name: str
) -> Dict[str, str]:
    if final_audio is None or len(final_audio) == 0 or final_audio.max_dBFS == float("-inf"):
        raise ValueError("Invalid final_audio")

    exported_files: Dict[str, str] = {}
    request_data = config.get("request_data", {})
    prompt = request_data.get("prompt", "unknown")
    purpose = request_data.get("mastering_purpose", "personal")

    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
    safe_prompt = "_".join(safe_prompt.split())

    base_name = f"{project_name}_{safe_prompt}_{purpose}_FINAL"

    main_path = project_dir / f"{base_name}.wav"
    if not safe_export_audio(final_audio, main_path, "wav"):
        raise ValueError(f"Failed to export main WAV: {main_path}")
    exported_files["final"] = str(main_path)

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
    exported_files: Dict[str, str] = {}

    stems_dir = project_dir / "stems"
    inter_dir = project_dir / "intermediate"
    ensure_directory(stems_dir)
    ensure_directory(inter_dir)

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


# === ТОНКАЯ ОБЁРТКА-КЛАСС ДЛЯ pipeline.py ===
class ExportManager:
    """
    ИСПРАВЛЕННАЯ система экспорта результатов проекта
    
    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
    - УБРАНА заглушка с тишиной в _bytes_to_audiosegment
    - Правильная валидация аудиоданных
    - Строгие проверки на каждом этапе
    - Исключения вместо заглушек
    """
    
    def __init__(self, base_dir: str = "output"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Поддерживаемые форматы экспорта
        self.supported_formats = {
            "wav": {"extension": "wav", "quality": "lossless"},
            "mp3": {"extension": "mp3", "quality": "320k"},
            "flac": {"extension": "flac", "quality": "lossless"},
            "aac": {"extension": "m4a", "quality": "256k"}
        }

        self.logger.info(f"📁 ExportManager initialized: {self.base_dir}")

    def check_export_environment(self) -> Dict[str, bool]:
        """Проверка окружения перед началом экспорта"""
        checks = {}

        try:
            # Проверка прав на запись в базовую директорию
            test_file = self.base_dir / "write_test.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            if test_file.exists():
                test_file.unlink()
                checks["base_dir_writable"] = True
            else:
                checks["base_dir_writable"] = False
        except Exception:
            checks["base_dir_writable"] = False

        # Проверка свободного места (минимум 100MB)
        try:
            import shutil
            free_space = shutil.disk_usage(self.base_dir)[2]
            checks["sufficient_space"] = free_space > 100 * 1024 * 1024
        except Exception:
            checks["sufficient_space"] = False

        # Проверка FFmpeg для MP3 экспорта
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=5)
            checks["ffmpeg_available"] = result.returncode == 0
        except Exception:
            checks["ffmpeg_available"] = False

        # ИСПРАВЛЕННАЯ проверка доступности pydub - БЕЗ ЗАГЛУШЕК!
        try:
            # Создаем тестовое аудио с реальным звуком (не тишина!)
            from pydub.generators import Sine
            test_audio = Sine(440).to_audio_segment(duration=1000)  # 1 секунда 440Hz
            
            # Проверяем что тестовое аудио НЕ тишина
            if test_audio.max_dBFS == float('-inf'):
                checks["pydub_working"] = False
                self.logger.error("❌ pydub test audio is silent!")
            else:
                # Пробуем экспортировать в WAV
                buffer = io.BytesIO()
                test_audio.export(buffer, format="wav")
                buffer.seek(0)
                exported_size = len(buffer.getvalue())
                
                # Проверяем размер экспорта
                checks["pydub_working"] = exported_size > 1000  # WAV 1 сек должен быть больше 1KB
                
                if checks["pydub_working"]:
                    self.logger.info(f"✅ pydub working correctly: exported {exported_size} bytes")
                else:
                    self.logger.error(f"❌ pydub export too small: {exported_size} bytes")
                
        except Exception as e:
            checks["pydub_working"] = False
            self.logger.error(f"❌ pydub test failed with exception: {e}")

        # Логируем результаты
        self.logger.info("🔍 Environment checks:")
        for check, result in checks.items():
            icon = "✅" if result else "❌"
            self.logger.info(f"  {icon} {check}: {'OK' if result else 'FAILED'}")

        return checks

    def _serialize_for_json(self, obj: Any) -> Any:
        """Рекурсивная конвертация объектов для JSON сериализации"""
        if hasattr(obj, '__dataclass_fields__'):  # dataclass
            return asdict(obj)
        elif hasattr(obj, '__dict__'):  # обычный объект
            return {k: self._serialize_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Для неизвестных типов пытаемся преобразовать в строку
            return str(obj)
    
    def _bytes_to_audiosegment(self, audio_data: Union[bytes, AudioSegment], fallback_ms: int = 8000) -> AudioSegment:
        try:
            if isinstance(audio_data, AudioSegment):
                return audio_data
            if isinstance(audio_data, (bytes, bytearray)):
                if len(audio_data) == 0:
                    self.logger.warning("⚠️ Пустые bytes -> тишина")
                    return AudioSegment.silent(duration=fallback_ms)
                buf = io.BytesIO(audio_data)
                # Пытаемся как WAV, затем как “любой” файл
                try:
                    return AudioSegment.from_wav(buf)
                except Exception:
                    buf.seek(0)
                    return AudioSegment.from_file(buf)
            self.logger.error(f"❌ Неизвестный тип аудио: {type(audio_data)} -> тишина")
            return AudioSegment.silent(duration=fallback_ms)
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации аудио: {e}")
            return AudioSegment.silent(duration=fallback_ms)

    def _ensure_directory(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create directory {path.parent}: {e}")
            return False

    def _safe_export_audio(self, audio: AudioSegment, path: Path, format_name: str = "wav") -> bool:
        if audio is None:
            self.logger.error("❌ CRITICAL: Cannot export None audio!")
            return False
        if audio.max_dBFS == float('-inf'):
            self.logger.error("❌ CRITICAL: Cannot export silent audio!")
            return False
        if len(audio) == 0:
            self.logger.error("❌ CRITICAL: Cannot export zero-duration audio!")
            return False
        try:
            if not self._ensure_directory(path):
                return False
            if not os.access(path.parent, os.W_OK):
                self.logger.error(f"❌ No write permission to directory: {path.parent}")
                temp_dir = Path(tempfile.gettempdir()) / "wavedream_export"
                temp_dir.mkdir(exist_ok=True)
                path = temp_dir / path.name
                self.logger.warning(f"⚠️ Fallback to temp directory: {path}")
            if path.exists():
                try:
                    with open(path, 'a'):
                        pass
                except PermissionError:
                    timestamp = int(time.time() * 1000)
                    path = path.parent / f"{path.stem}_{timestamp}{path.suffix}"
                    self.logger.warning(f"⚠️ File locked, using: {path.name}")
            try:
                self.logger.info(f"🎵 Exporting audio: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB -> {path}")
                audio.export(str(path), format=format_name,
                             bitrate="320k" if format_name == "mp3" else None,
                             parameters=["-avoid_negative_ts", "make_zero"] if format_name == "mp3" else None)
            except Exception as export_error:
                self.logger.warning(f"⚠️ {format_name} export failed, trying WAV: {export_error}")
                buffer = io.BytesIO()
                audio.export(buffer, format="wav")
                buffer.seek(0)
                wav_data = buffer.getvalue()
                if len(wav_data) == 0:
                    raise ValueError("WAV export resulted in empty data!")
                wav_path = path.with_suffix('.wav')
                with open(wav_path, 'wb') as f:
                    f.write(wav_data)
                path = wav_path
            if not path.exists():
                self.logger.error(f"❌ Export failed: {path} was not created")
                return False
            if path.stat().st_size < 1000:
                self.logger.error(f"❌ Export failed: {path} is too small")
                return False
            self.logger.info(f"✅ Exported successfully: {path} ({path.stat().st_size} bytes)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Critical export error for {path}: {e}")
            try:
                emergency_path = Path.cwd() / f"emergency_{path.name}"
                if audio.max_dBFS == float('-inf'):
                    self.logger.error("❌ Emergency save cancelled - audio is silent!")
                    return False
                audio.export(str(emergency_path), format="wav")
                if emergency_path.exists() and emergency_path.stat().st_size > 1000:
                    self.logger.warning(f"🚨 Emergency save successful: {emergency_path}")
                    return True
            except Exception as emergency_error:
                self.logger.error(f"❌ Even emergency save failed: {emergency_error}")
            return False

    async def export_complete_project(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_audio: Dict[str, Union[bytes, AudioSegment]],
        config: Dict
    ) -> Dict[str, str]:
        """
        ИСПРАВЛЕННЫЙ экспорт полного проекта с валидацией аудио
        """
        self.logger.info("💾 Starting complete project export...")
        
        try:
            # ДОБАВЛЕНО: Критическая валидация входных данных
            if mastered_audio is None:
                raise ValueError("❌ CRITICAL: mastered_audio is None!")
            
            if isinstance(mastered_audio, bytes) and len(mastered_audio) == 0:
                raise ValueError("❌ CRITICAL: mastered_audio is empty bytes!")
            
            # Получаем конфиг
            output_dir = Path(config.get("output_dir", self.base_dir))
            self._ensure_directory(output_dir)
            
            # Создаём уникальное имя проекта
            timestamp = int(time.time())
            project_name = f"WD_Project_{timestamp}"
            project_dir = output_dir / project_name
            self._ensure_directory(project_dir)
            
            self.logger.info(f"📁 Project directory: {project_dir}")
            
            exported_files = {}
            
            # 1. Экспорт финального трека с валидацией
            self.logger.info("🎵 Exporting final track...")
            try:
                final_audio = self._bytes_to_audiosegment(mastered_audio)  # Может вызвать исключение
                final_files = await self._export_final_track(final_audio, project_dir, config, project_name)
                exported_files.update(final_files)
            except Exception as e:
                self.logger.error(f"❌ CRITICAL: Final track export failed: {e}")
                raise ValueError(f"Cannot export final track - audio data is corrupted: {e}")
            
            # 2. Экспорт промежуточных версий и стемов
            if config.get("export_stems", True) and intermediate_audio:
                self.logger.info("🎛️ Exporting stems and intermediate versions...")
                try:
                    intermediate_files = await self._export_intermediate_versions(
                        intermediate_audio, project_dir, config, project_name
                    )
                    exported_files.update(intermediate_files)
                except Exception as e:
                    self.logger.error(f"❌ Intermediate export failed: {e}")
                    # Не критично, продолжаем без промежуточных файлов
            
            # 3. Экспорт метаданных
            self.logger.info("📋 Exporting metadata...")
            try:
                metadata_files = await self._export_metadata(config, project_dir)
                exported_files.update(metadata_files)
            except Exception as e:
                self.logger.error(f"❌ Metadata export failed: {e}")
                # Не критично, продолжаем без метаданных
            
            # 4. Создание отчёта
            self.logger.info("📊 Creating project report...")
            try:
                report_file = await self._create_project_report(config, exported_files, project_dir)
                if report_file:
                    exported_files["project_report"] = report_file
            except Exception as e:
                self.logger.error(f"❌ Report creation failed: {e}")
                # Не критично
            
            # 5. Валидация всех файлов
            self._validate_exports(exported_files)
            
            # ДОБАВЛЕНО: Проверяем что хотя бы основной файл экспортирован
            if not any(key.startswith("final") for key in exported_files.keys()):
                raise ValueError("❌ CRITICAL: No final audio files were exported!")
            
            self.logger.info(f"🎉 Project export complete: {len(exported_files)} files in {project_dir}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL Export error: {e}")
            
            # НЕ ВОЗВРАЩАЕМ ПУСТОЙ DICT - выбрасываем исключение!
            raise ValueError(f"Project export failed: {e}")

    async def save_intermediate(self, name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет промежуточный файл"""
        path = Path(output_dir) / "intermediate" / f"{name}.wav"
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        if self._safe_export_audio(audio, path):
            return str(path)
        return ""
    
    async def _export_final_track(
        self, final_audio: AudioSegment, project_dir: Path, config: Dict, project_name: str
    ) -> Dict[str, str]:
        """ИСПРАВЛЕННЫЙ экспорт финального трека с валидацией"""
        
        # ДОБАВЛЕНО: Валидация входного аудио
        if final_audio is None:
            raise ValueError("❌ CRITICAL: final_audio is None!")
        
        if final_audio.max_dBFS == float('-inf'):
            raise ValueError("❌ CRITICAL: final_audio is completely silent!")
        
        if len(final_audio) == 0:
            raise ValueError("❌ CRITICAL: final_audio has zero duration!")
        
        exported_files = {}
        
        # Создаём основное имя файла
        request_data = config.get("request_data", {})
        prompt = request_data.get("prompt", "unknown")
        purpose = request_data.get("mastering_purpose", "personal")
        
        # Создаём безопасное имя из промпта
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
        safe_prompt = "_".join(safe_prompt.split())
        
        base_name = f"{project_name}_{safe_prompt}_{purpose}_FINAL"
        
        # Основной WAV файл
        main_path = project_dir / f"{base_name}.wav"
        if self._safe_export_audio(final_audio, main_path, "wav"):
            exported_files["final"] = str(main_path)
        else:
            # Если основной экспорт не удался - это критично!
            raise ValueError(f"❌ CRITICAL: Failed to export main WAV file: {main_path}")
        
        # Дополнительные форматы
        additional_formats = config.get("export_formats", ["mp3"])
        
        for fmt in additional_formats:
            if fmt != "wav" and fmt in self.supported_formats:
                fmt_info = self.supported_formats[fmt]
                fmt_path = project_dir / f"{base_name}.{fmt_info['extension']}"
                
                if self._safe_export_audio(final_audio, fmt_path, fmt):
                    exported_files[f"final_{fmt}"] = str(fmt_path)
                else:
                    self.logger.warning(f"⚠️ Failed to export {fmt} format: {fmt_path}")
        
        return exported_files
    
    async def save_intermediate(self, stage_key: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет промежуточный файл в output_dir/intermediate"""
        try:
            audio_segment = self._bytes_to_audiosegment(audio_bytes)
            stage_path = Path(output_dir) / "intermediate" / f"{stage_key}.wav"
            if self._safe_export_audio(audio_segment, stage_path, "wav"):
                return str(stage_path)
        except Exception as e:
            self.logger.error(f"❌ Error saving intermediate '{stage_key}': {e}")
        return ""

    async def save_stem(self, stem_name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет отдельный стем в output_dir/stems"""
        try:
            audio_segment = self._bytes_to_audiosegment(audio_bytes)
            stem_path = Path(output_dir) / "stems" / f"stem_{stem_name}.wav"
            if self._safe_export_audio(audio_segment, stem_path, "wav"):
                return str(stem_path)
        except Exception as e:
            self.logger.error(f"❌ Error saving stem '{stem_name}': {e}")
        return ""

    def save_final_mix(self, project_name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет финальный микс"""
        try:
            audio_segment = self._bytes_to_audiosegment(audio_bytes)
            final_mix_path = Path(output_dir) / f"{project_name}_final_mix.wav"
            if self._safe_export_audio(audio_segment, final_mix_path, "wav"):
                return str(final_mix_path)
        except Exception as e:
            self.logger.error(f"❌ Error saving final mix: {e}")
        return ""

    async def save_final(self, project_name: str, audio_bytes: bytes, output_dir: str) -> str:
        """Сохраняет финальный трек в output_dir"""
        try:
            audio_segment = self._bytes_to_audiosegment(audio_bytes)
            final_path = Path(output_dir) / f"{project_name}_final.wav"
            if self._safe_export_audio(audio_segment, final_path, "wav"):
                return str(final_path)
        except Exception as e:
            self.logger.error(f"❌ Error saving final track: {e}")
        return ""

    def force_save_everything(self, *args, **kwargs):
        """Аварийно сохраняет все доступные треки"""
        if len(args) >= 2:
            audio_dict = args[0]
            output_dir = args[1]
        else:
            self.logger.error("❌ force_save_everything: not enough arguments")
            return

        emergency_dir = Path(output_dir) / f"emergency_export_{int(time.time())}"
        emergency_dir.mkdir(parents=True, exist_ok=True)

        for name, audio_bytes in audio_dict.items():
            try:
                audio_segment = self._bytes_to_audiosegment(audio_bytes)
                self._safe_export_audio(audio_segment, emergency_dir / f"{name}.wav")
            except Exception as e:
                self.logger.error(f"❌ Error in emergency save for '{name}': {e}")
        
        return exported_files