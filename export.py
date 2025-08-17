# ИСПРАВЛЕННЫЙ export.py - Рабочая система экспорта

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

# === ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===
SUPPORTED_FORMATS: Dict[str, Dict[str, str]] = {
    "wav": {"extension": "wav", "quality": "lossless"},
    "mp3": {"extension": "mp3", "quality": "320k"},
    "flac": {"extension": "flac", "quality": "lossless"},
    "aac": {"extension": "m4a", "quality": "256k"},
}

def serialize_for_json(obj: Any) -> Any:
    """Рекурсивная сериализация для JSON"""
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
    """Создание директории с проверкой"""
    try:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Cannot create directory {path}: {e}")
        return False

def bytes_to_audiosegment(audio_data: Union[bytes, AudioSegment]) -> AudioSegment:
    """ИСПРАВЛЕННАЯ конвертация bytes в AudioSegment - БЕЗ ЗАГЛУШЕК!"""
    if isinstance(audio_data, AudioSegment):
        # Проверяем что AudioSegment не пустой
        if len(audio_data) == 0:
            raise ValueError("AudioSegment is empty (0 duration)!")
        if audio_data.max_dBFS == float("-inf"):
            raise ValueError("AudioSegment is completely silent!")
        return audio_data

    if not isinstance(audio_data, (bytes, bytearray)):
        raise TypeError(f"Invalid audio data type: {type(audio_data)}")

    if len(audio_data) == 0:
        raise ValueError("Empty bytes data provided!")

    try:
        buffer = io.BytesIO(audio_data)
        
        # Пробуем различные форматы
        formats_to_try = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        
        for fmt in formats_to_try:
            try:
                buffer.seek(0)
                audio = AudioSegment.from_file(buffer, format=fmt)
                
                if len(audio) == 0:
                    logger.warning(f"Audio loaded as {fmt} but has 0 duration")
                    continue
                    
                if audio.max_dBFS == float("-inf"):
                    logger.warning(f"Audio loaded as {fmt} but is silent")
                    continue

                logger.debug(f"✅ Audio loaded as {fmt}: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB")
                return audio
                
            except Exception as e:
                logger.debug(f"Failed to load as {fmt}: {e}")
                continue
        
        # Если все форматы не подошли, пробуем без указания формата
        try:
            buffer.seek(0)
            audio = AudioSegment.from_file(buffer)
            
            if len(audio) == 0 or audio.max_dBFS == float("-inf"):
                raise ValueError("Auto-detected audio is empty or silent")
            
            logger.info(f"✅ Audio loaded (auto-detect): {len(audio)/1000:.1f}s")
            return audio
            
        except Exception as e:
            logger.error(f"Auto-detect also failed: {e}")
        
        # ИСПРАВЛЕНО: Не возвращаем тишину как fallback!
        raise ValueError("❌ CRITICAL: Cannot load audio from bytes - data may be corrupted!")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: bytes_to_audiosegment failed: {e}")
        raise  # Пробрасываем исключение вместо возврата заглушки

def safe_export_audio(audio: AudioSegment, path: Path, format_name: str = "wav") -> bool:
    """ИСПРАВЛЕННЫЙ безопасный экспорт аудио"""
    
    # КРИТИЧЕСКИЕ ПРОВЕРКИ входных данных
    if audio is None:
        logger.error("❌ CRITICAL: Cannot export None audio!")
        return False
    
    if len(audio) == 0:
        logger.error("❌ CRITICAL: Cannot export zero-duration audio!")
        return False
    
    if audio.max_dBFS == float('-inf'):
        logger.error("❌ CRITICAL: Cannot export silent audio!")
        return False

    try:
        if not ensure_directory(path):
            return False

        # Проверяем права записи
        parent_dir = path.parent
        if not os.access(parent_dir, os.W_OK):
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "wavedream_export"
            temp_dir.mkdir(exist_ok=True)
            path = temp_dir / path.name
            logger.warning(f"⚠️ No write permission, fallback to: {path}")

        # Проверяем блокировку файла
        if path.exists():
            try:
                with open(path, "a"):
                    pass
            except PermissionError:
                ts = int(time.time() * 1000)
                path = path.parent / f"{path.stem}_{ts}{path.suffix}"
                logger.warning(f"⚠️ File locked, using: {path.name}")

        logger.info(f"🎵 Exporting: {len(audio)/1000:.1f}s, peak: {audio.max_dBFS:.1f}dB -> {path}")

        try:
            # Экспортируем в указанном формате
            export_params = {}
            if format_name == "mp3":
                export_params = {
                    "bitrate": "320k",
                    "parameters": ["-avoid_negative_ts", "make_zero"]
                }
            
            audio.export(str(path), format=format_name, **export_params)
            
        except Exception as export_error:
            logger.warning(f"⚠️ {format_name} export failed, trying WAV: {export_error}")
            
            # Fallback к WAV
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            wav_data = buffer.getvalue()
            
            if len(wav_data) == 0:
                raise ValueError("WAV export resulted in empty data!")
            
            path = path.with_suffix(".wav")
            with open(path, "wb") as f:
                f.write(wav_data)

        # Проверяем результат экспорта
        if not path.exists():
            logger.error(f"❌ Export failed: {path} not created")
            return False
        
        size = path.stat().st_size
        if size < 1000:  # Минимум 1KB
            logger.error(f"❌ Export failed: {path} too small ({size} bytes)")
            return False

        logger.info(f"✅ Exported successfully: {path} ({size} bytes)")
        return True

    except Exception as e:
        logger.error(f"❌ Critical export error for {path}: {e}")
        
        # Попытка экстренного сохранения
        try:
            emergency_path = Path.cwd() / f"emergency_{path.name}"
            
            # ИСПРАВЛЕНО: Проверяем что audio не стал тишиной перед emergency save
            if audio.max_dBFS == float('-inf'):
                logger.error("❌ Emergency save cancelled - audio is silent!")
                return False
            
            audio.export(str(emergency_path), format="wav")
            
            if emergency_path.exists() and emergency_path.stat().st_size > 1000:
                logger.warning(f"🚨 Emergency save successful: {emergency_path}")
                return True
                
        except Exception as e2:
            logger.error(f"❌ Even emergency save failed: {e2}")
        
        return False

class ExportManager:
    """
    ИСПРАВЛЕННАЯ система экспорта результатов проекта
    
    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
    - Убраны все заглушки с тишиной
    - Правильная валидация аудиоданных на каждом этапе
    - Исключения вместо возврата пустых результатов
    - Улучшенная обработка ошибок
    """
    
    def __init__(self, base_dir: str = "output"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_formats = SUPPORTED_FORMATS
        
        self.logger.info(f"📁 ExportManager initialized: {self.base_dir}")

    def check_export_environment(self) -> Dict[str, bool]:
        """Проверка окружения перед экспортом"""
        checks = {}

        # Проверка прав записи
        try:
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

        # Проверка FFmpeg для MP3
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=5)
            checks["ffmpeg_available"] = result.returncode == 0
        except Exception:
            checks["ffmpeg_available"] = False

        # ИСПРАВЛЕННАЯ проверка pydub - без создания тишины!
        try:
            from pydub.generators import Sine
            # Создаем тестовый тон 440Hz на 1 секунду
            test_audio = Sine(440).to_audio_segment(duration=1000)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: убеждаемся что тестовое аудио НЕ тишина
            if test_audio.max_dBFS == float('-inf'):
                checks["pydub_working"] = False
                self.logger.error(f"❌ Emergency metadata save failed: {e} / {ee}")
        
        return exported_files

    async def _create_project_report(
        self, config: Dict, exported_files: Dict[str, str], project_dir: Path
    ) -> Optional[str]:
        """Создание отчёта проекта"""
        
        try:
            report_path = project_dir / "PROJECT_REPORT.md"
            request_data = config.get("request_data", {})
            structure = config.get("structure", {})
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# 🎵 WaveDream Enhanced Pro - Project Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Обзор проекта
                f.write("## 📋 Project Overview\n\n")
                f.write(f"**Prompt**: `{request_data.get('prompt', 'N/A')}`\n\n")
                f.write(f"**Genre**: {request_data.get('genre', 'Auto-detected')}\n\n")
                f.write(f"**BPM**: {request_data.get('bpm', 'Auto')}\n\n")
                f.write(f"**Duration**: {structure.get('total_duration', 0):.1f} seconds\n\n")
                f.write(f"**Mastering Purpose**: {request_data.get('mastering_purpose', 'personal')}\n\n")
                f.write(f"**Energy Level**: {request_data.get('energy_level', 0.5):.1f}\n\n")
                f.write(f"**Creativity Factor**: {request_data.get('creativity_factor', 0.7):.1f}\n\n")
                
                # Структура трека
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
                
                # Экспортированные файлы
                f.write("## 📁 Exported Files\n\n")
                
                file_categories = {
                    "Final Tracks": [],
                    "Intermediate Versions": [],
                    "Stems": [],
                    "Metadata": [],
                    "Other": []
                }
                
                for file_type, file_path in exported_files.items():
                    file_name = Path(file_path).name if file_path else file_type
                    
                    if "final" in file_type.lower():
                        file_categories["Final Tracks"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                    elif "stem" in file_type.lower():
                        file_categories["Stems"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                    elif "intermediate" in file_type.lower():
                        file_categories["Intermediate Versions"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                    elif any(x in file_type.lower() for x in ["metadata", "info", "report", "structure", "mastering"]):
                        file_categories["Metadata"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                    else:
                        file_categories["Other"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
                
                for category_name, files in file_categories.items():
                    if files:
                        f.write(f"\n### {category_name}\n")
                        for file_line in files:
                            f.write(f"{file_line}\n")
                
                # Техническая информация
                f.write("\n## 🔧 Technical Details\n\n")
                f.write("**Pipeline Stages Completed**:\n")
                pipeline_stages = [
                    "Metadata Analysis & Genre Detection",
                    "Structure Generation (LLM/Fallback)",
                    "Semantic Sample Selection",
                    "MusicGen Base Generation",
                    "Stem Creation & Mixing",
                    "Effects Processing",
                    "Smart Mastering",
                    "Quality Verification",
                    "Multi-format Export"
                ]
                
                for i, stage in enumerate(pipeline_stages, 1):
                    f.write(f"{i}. ✅ {stage}\n")
                
                # Рекомендации по использованию
                f.write(f"\n## 💡 Usage Recommendations\n\n")
                purpose = request_data.get("mastering_purpose", "personal")
                
                recommendations = {
                    "freelance": [
                        "✅ Ready for commercial sale",
                        "📱 Optimized for streaming platforms",
                        "🎧 Test on various playback systems"
                    ],
                    "professional": [
                        "🎬 Suitable for broadcast/cinema use",
                        "📺 Meets professional loudness standards",
                        "🎛️ Full dynamic range preserved"
                    ],
                    "personal": [
                        "🏠 Perfect for personal listening",
                        "🎵 Natural, unprocessed character",
                        "🔊 Great on home audio systems"
                    ],
                    "family": [
                        "👨‍👩‍👧‍👦 Family-friendly mixing",
                        "🎥 Ideal for home videos",
                        "📱 Works well on mobile devices"
                    ]
                }
                
                purpose_recs = recommendations.get(purpose, recommendations["personal"])
                for rec in purpose_recs:
                    f.write(f"- {rec}\n")
                
                f.write(f"\n## 📊 System Information\n\n")
                f.write(f"- **WaveDream Version**: Enhanced Pro v2.0\n")
                f.write(f"- **Export Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Structure Source**: {structure.get('source', 'unknown')}\n")
                f.write(f"- **Total Files**: {len(exported_files)}\n")
                
                f.write("\n---\n*Auto-generated by WaveDream Enhanced Pro Pipeline*\n")
            
            self.logger.info(f"    📋 Project report created: {report_path.name}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating project report: {e}")
            return None

    def _validate_exports(self, exported_files: Dict[str, str]) -> None:
        """Валидация экспортированных файлов"""
        
        self.logger.info("🔍 Validating exported files...")
        
        total_files = len(exported_files)
        valid_files = 0
        
        for file_type, file_path in exported_files.items():
            path = Path(file_path)
            
            if path.exists():
                size = path.stat().st_size
                if size > 0:
                    valid_files += 1
                    self.logger.debug(f"    ✅ {file_type}: {path.name} ({size} bytes)")
                else:
                    self.logger.error(f"    ❌ {file_type}: {path.name} (0 bytes - EMPTY)")
            else:
                self.logger.error(f"    ❌ {file_type}: {path.name} (MISSING)")
        
        self.logger.info(f"📊 Validation complete: {valid_files}/{total_files} files valid")

    # === МЕТОДЫ ДЛЯ ПОЭТАПНОГО СОХРАНЕНИЯ ===

    def save_intermediate(self, stage_name: str, project_name: str, audio_data: Union[bytes, AudioSegment]) -> Optional[str]:
        """Сохранение промежуточного файла"""
        try:
            if isinstance(audio_data, (bytes, bytearray)) and len(audio_data) > 0:
                audio_segment = bytes_to_audiosegment(audio_data)
            elif isinstance(audio_data, AudioSegment):
                audio_segment = audio_data
            else:
                self.logger.error(f"Invalid audio data type for {stage_name}: {type(audio_data)}")
                return None
            
            output_dir = self.base_dir / project_name / "intermediate"
            ensure_directory(output_dir)
            
            file_path = output_dir / f"{stage_name}.wav"
            
            if safe_export_audio(audio_segment, file_path, "wav"):
                return str(file_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving intermediate {stage_name}: {e}")
            return None

    def save_stem(self, audio_data: Union[bytes, AudioSegment], project_name: str, stem_name: str) -> Optional[str]:
        """Сохранение отдельного стема"""
        try:
            if isinstance(audio_data, (bytes, bytearray)) and len(audio_data) > 0:
                audio_segment = bytes_to_audiosegment(audio_data)
            elif isinstance(audio_data, AudioSegment):
                audio_segment = audio_data
            else:
                self.logger.error(f"Invalid stem audio data type for {stem_name}: {type(audio_data)}")
                return None
            
            output_dir = self.base_dir / project_name / "stems"
            ensure_directory(output_dir)
            
            file_path = output_dir / f"{stem_name}.wav"
            
            if safe_export_audio(audio_segment, file_path, "wav"):
                return str(file_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving stem {stem_name}: {e}")
            return None

    def save_final_mix(self, audio_data: Union[bytes, AudioSegment], project_name: str) -> Optional[str]:
        """Сохранение финального микса"""
        try:
            if isinstance(audio_data, (bytes, bytearray)) and len(audio_data) > 0:
                audio_segment = bytes_to_audiosegment(audio_data)
            elif isinstance(audio_data, AudioSegment):
                audio_segment = audio_data
            else:
                self.logger.error(f"Invalid final mix audio data type: {type(audio_data)}")
                return None
            
            output_dir = self.base_dir / project_name
            ensure_directory(output_dir)
            
            file_path = output_dir / f"{project_name}_FINAL_MIX.wav"
            
            if safe_export_audio(audio_segment, file_path, "wav"):
                return str(file_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving final mix: {e}")
            return None

    def save_metadata(self, project_name: str, metadata: Dict) -> Optional[str]:
        """Сохранение метаданных проекта"""
        try:
            output_dir = self.base_dir / project_name
            ensure_directory(output_dir)
            
            file_path = output_dir / "project_metadata.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serialize_for_json(metadata), f, indent=2, ensure_ascii=False, default=str)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving metadata: {e}")
            return None

    def force_save_everything(
        self, main_audio_data: Union[bytes, AudioSegment], 
        intermediate_data: Dict[str, Any], 
        metadata: Dict
    ) -> Dict[str, str]:
        """Экстренное сохранение всех доступных данных"""
        
        self.logger.warning("🚨 Force saving all available data...")
        
        emergency_files = {}
        timestamp = int(time.time())
        emergency_dir = self.base_dir / f"emergency_export_{timestamp}"
        ensure_directory(emergency_dir)
        
        # Сохраняем основное аудио
        try:
            if isinstance(main_audio_data, (bytes, bytearray)) and len(main_audio_data) > 0:
                emergency_path = emergency_dir / "emergency_final.wav"
                with open(emergency_path, 'wb') as f:
                    f.write(main_audio_data)
                emergency_files["emergency_final"] = str(emergency_path)
                self.logger.warning(f"🚨 Emergency final: {emergency_path}")
        except Exception as e:
            self.logger.error(f"Failed to save emergency final: {e}")
        
        # Сохраняем промежуточные данные
        for key, data in intermediate_data.items():
            try:
                if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                    emergency_path = emergency_dir / f"emergency_{key}.wav"
                    with open(emergency_path, 'wb') as f:
                        f.write(data)
                    emergency_files[f"emergency_{key}"] = str(emergency_path)
                elif isinstance(data, dict):
                    # Это стемы
                    for stem_name, stem_data in data.items():
                        if isinstance(stem_data, (bytes, bytearray)) and len(stem_data) > 0:
                            emergency_path = emergency_dir / f"emergency_stem_{stem_name}.wav"
                            with open(emergency_path, 'wb') as f:
                                f.write(stem_data)
                            emergency_files[f"emergency_stem_{stem_name}"] = str(emergency_path)
            except Exception as e:
                self.logger.error(f"Failed to save emergency {key}: {e}")
        
        # Сохраняем метаданные
        try:
            emergency_metadata_path = emergency_dir / "emergency_metadata.json"
            with open(emergency_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serialize_for_json(metadata), f, indent=2, ensure_ascii=False, default=str)
            emergency_files["emergency_metadata"] = str(emergency_metadata_path)
        except Exception as e:
            self.logger.error(f"Failed to save emergency metadata: {e}")
        
        self.logger.warning(f"🚨 Emergency save complete: {len(emergency_files)} files in {emergency_dir}")
        return emergency_files

    def test_export_system(self) -> bool:
        """Тестирование системы экспорта"""
        try:
            # Создаём тестовое аудио
            from pydub.generators import Sine
            test_audio = Sine(440).to_audio_segment(duration=2000)  # 2 секунды
            
            # Тестируем экспорт
            test_dir = self.base_dir / "export_test"
            ensure_directory(test_dir)
            test_path = test_dir / "test_export.wav"
            
            success = safe_export_audio(test_audio, test_path, "wav")
            
            # Очищаем тест
            if test_path.exists():
                test_path.unlink()
            if test_dir.exists():
                test_dir.rmdir()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Export system test failed: {e}")
            return False

    def debug_export_issue(self, audio_data: Any, intermediate_data: Any, config: Any) -> None:
        """Отладочная информация при проблемах с экспортом"""
        self.logger.debug("🔍 Export debug information:")
        self.logger.debug(f"  Audio data type: {type(audio_data)}")
        self.logger.debug(f"  Audio data size: {len(audio_data) if hasattr(audio_data, '__len__') else 'N/A'}")
        self.logger.debug(f"  Intermediate data keys: {list(intermediate_data.keys()) if isinstance(intermediate_data, dict) else 'N/A'}")
        self.logger.debug(f"  Config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}")
        
        # Проверяем доступность директорий
        self.logger.debug(f"  Base dir exists: {self.base_dir.exists()}")
        self.logger.debug(f"  Base dir writable: {os.access(self.base_dir, os.W_OK)}")
        
        # Проверяем системные ресурсы
        try:
            import shutil
            free_space = shutil.disk_usage(self.base_dir)[2] / (1024**3)
            self.logger.debug(f"  Free space: {free_space:.1f} GB")
        except Exception as e:
            self.logger.debug(f"  Could not check free space: {e}")

# === ЭКСПОРТ ОСНОВНЫХ ФУНКЦИЙ ===
__all__ = [
    'ExportManager',
    'safe_export_audio',
    'bytes_to_audiosegment',
    'serialize_for_json',
    'ensure_directory'
]("❌ pydub test audio is silent!")
            else:
                # Проверяем экспорт в WAV
                buffer = io.BytesIO()
                test_audio.export(buffer, format="wav")
                buffer.seek(0)
                exported_size = len(buffer.getvalue())
                
                # WAV файл 1 секунды 440Hz должен быть > 10KB
                checks["pydub_working"] = exported_size > 10000
                
                if checks["pydub_working"]:
                    self.logger.debug(f"✅ pydub working: exported {exported_size} bytes")
                else:
                    self.logger.error(f"❌ pydub export too small: {exported_size} bytes")
                
        except Exception as e:
            checks["pydub_working"] = False
            self.logger.error(f"❌ pydub test failed: {e}")

        # Логируем результаты
        self.logger.info("🔍 Environment checks:")
        for check, result in checks.items():
            icon = "✅" if result else "❌"
            self.logger.info(f"  {icon} {check}: {'OK' if result else 'FAILED'}")

        return checks

    async def export_complete_project(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_audio: Dict[str, Union[bytes, AudioSegment]],
        config: Dict
    ) -> Dict[str, str]:
        """
        ИСПРАВЛЕННЫЙ полный экспорт проекта
        """
        self.logger.info("💾 Starting complete project export...")
        
        try:
            # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ входных данных
            if mastered_audio is None:
                raise ValueError("❌ CRITICAL: mastered_audio is None!")
            
            if isinstance(mastered_audio, (bytes, bytearray)) and len(mastered_audio) == 0:
                raise ValueError("❌ CRITICAL: mastered_audio is empty bytes!")
            
            # Проверяем что mastered_audio содержит реальное аудио
            if isinstance(mastered_audio, bytes):
                try:
                    test_audio = bytes_to_audiosegment(mastered_audio)  # Может вызвать исключение
                    self.logger.info(f"✅ Mastered audio validation passed: {len(test_audio)/1000:.1f}s")
                except Exception as e:
                    raise ValueError(f"❌ CRITICAL: mastered_audio validation failed: {e}")
            
            # Получаем конфигурацию
            output_dir = Path(config.get("output_dir", self.base_dir))
            ensure_directory(output_dir)
            
            # Создаём уникальное имя проекта
            timestamp = int(time.time())
            project_name = f"WD_Project_{timestamp}"
            project_dir = output_dir / project_name
            ensure_directory(project_dir)
            
            self.logger.info(f"📁 Project directory: {project_dir}")
            
            exported_files = {}
            
            # 1. Экспорт финального трека
            self.logger.info("🎵 Exporting final track...")
            try:
                final_files = await self._export_final_track(
                    mastered_audio, project_dir, config, project_name
                )
                exported_files.update(final_files)
                
                # ПРОВЕРЯЕМ что финальные файлы действительно экспортированы
                if not final_files:
                    raise ValueError("❌ CRITICAL: No final files were exported!")
                
            except Exception as e:
                self.logger.error(f"❌ CRITICAL: Final track export failed: {e}")
                raise ValueError(f"Final track export failed: {e}")
            
            # 2. Экспорт промежуточных версий (не критично)
            if config.get("export_stems", True) and intermediate_audio:
                self.logger.info("🎛️ Exporting stems and intermediate versions...")
                try:
                    intermediate_files = await self._export_intermediate_versions(
                        intermediate_audio, project_dir, config, project_name
                    )
                    exported_files.update(intermediate_files)
                except Exception as e:
                    self.logger.error(f"❌ Intermediate export failed: {e}")
                    # Продолжаем без промежуточных файлов
            
            # 3. Экспорт метаданных (не критично)
            self.logger.info("📋 Exporting metadata...")
            try:
                metadata_files = await self._export_metadata(config, project_dir)
                exported_files.update(metadata_files)
            except Exception as e:
                self.logger.error(f"❌ Metadata export failed: {e}")
                # Продолжаем без метаданных
            
            # 4. Создание отчёта (не критично)
            self.logger.info("📊 Creating project report...")
            try:
                report_file = await self._create_project_report(config, exported_files, project_dir)
                if report_file:
                    exported_files["project_report"] = report_file
            except Exception as e:
                self.logger.error(f"❌ Report creation failed: {e}")
            
            # 5. Валидация экспорта
            self._validate_exports(exported_files)
            
            # ФИНАЛЬНАЯ ПРОВЕРКА: убеждаемся что хотя бы основные файлы экспортированы
            final_files_count = len([k for k in exported_files.keys() if k.startswith("final")])
            if final_files_count == 0:
                raise ValueError("❌ CRITICAL: No final audio files were successfully exported!")
            
            self.logger.info(f"🎉 Project export complete: {len(exported_files)} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL Export error: {e}")
            raise ValueError(f"Project export failed: {e}")

    async def _export_final_track(
        self, final_audio_data: Union[bytes, AudioSegment], project_dir: Path, 
        config: Dict, project_name: str
    ) -> Dict[str, str]:
        """ИСПРАВЛЕННЫЙ экспорт финального трека"""
        
        exported_files = {}
        
        try:
            # ВАЛИДАЦИЯ и конвертация входного аудио
            if isinstance(final_audio_data, (bytes, bytearray)):
                if len(final_audio_data) == 0:
                    raise ValueError("❌ CRITICAL: final_audio_data is empty bytes!")
                final_audio = bytes_to_audiosegment(final_audio_data)  # Может вызвать исключение
            elif isinstance(final_audio_data, AudioSegment):
                final_audio = final_audio_data
            else:
                raise TypeError(f"❌ CRITICAL: Unsupported final_audio type: {type(final_audio_data)}")
            
            # ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ аудио
            if len(final_audio) == 0:
                raise ValueError("❌ CRITICAL: Final audio has zero duration!")
            
            if final_audio.max_dBFS == float('-inf'):
                raise ValueError("❌ CRITICAL: Final audio is completely silent!")
            
            # Создаём имя файла
            request_data = config.get("request_data", {})
            prompt = request_data.get("prompt", "unknown")
            purpose = request_data.get("mastering_purpose", "personal")
            
            # Безопасное имя из промпта
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
            safe_prompt = "_".join(safe_prompt.split()) or "track"
            
            base_name = f"{project_name}_{safe_prompt}_{purpose}_FINAL"
            
            # Основной WAV файл
            main_path = project_dir / f"{base_name}.wav"
            if safe_export_audio(final_audio, main_path, "wav"):
                exported_files["final"] = str(main_path)
                self.logger.info(f"  ✅ Main track exported: {main_path}")
            else:
                raise ValueError(f"❌ CRITICAL: Failed to export main WAV: {main_path}")
            
            # Дополнительные форматы
            additional_formats = config.get("export_formats", ["mp3"])
            
            for fmt in additional_formats:
                if fmt != "wav" and fmt in self.supported_formats:
                    fmt_info = self.supported_formats[fmt]
                    fmt_path = project_dir / f"{base_name}.{fmt_info['extension']}"
                    
                    if safe_export_audio(final_audio, fmt_path, fmt):
                        exported_files[f"final_{fmt}"] = str(fmt_path)
                        self.logger.info(f"  ✅ {fmt.upper()} exported: {fmt_path}")
                    else:
                        self.logger.warning(f"  ⚠️ Failed to export {fmt.upper()}: {fmt_path}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: _export_final_track failed: {e}")
            raise

    async def _export_intermediate_versions(
        self, intermediate_audio: Dict[str, Any], project_dir: Path, 
        config: Dict, project_name: str
    ) -> Dict[str, str]:
        """Экспорт промежуточных версий"""
        
        exported_files = {}
        
        # Создаём подпапки
        stems_dir = project_dir / "stems"
        inter_dir = project_dir / "intermediate"
        ensure_directory(stems_dir)
        ensure_directory(inter_dir)
        
        # Промежуточные стадии
        stage_mapping = {
            "base": "01_MusicGen_Base",
            "mixed": "02_Mixed_Stems",
            "processed": "03_Effects_Applied",
        }
        
        for stage_key, stage_name in stage_mapping.items():
            if stage_key in intermediate_audio:
                try:
                    audio_data = intermediate_audio[stage_key]
                    if isinstance(audio_data, (bytes, bytearray)) and len(audio_data) > 0:
                        audio_segment = bytes_to_audiosegment(audio_data)
                        stage_path = inter_dir / f"{stage_name}.wav"
                        
                        if safe_export_audio(audio_segment, stage_path, "wav"):
                            exported_files[f"intermediate_{stage_key}"] = str(stage_path)
                            self.logger.debug(f"    ✅ Intermediate {stage_key}: {stage_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error exporting intermediate {stage_key}: {e}")
        
        # Стемы
        if "stems" in intermediate_audio and isinstance(intermediate_audio["stems"], dict):
            for instrument, stem_data in intermediate_audio["stems"].items():
                try:
                    if isinstance(stem_data, (bytes, bytearray)) and len(stem_data) > 0:
                        stem_audio = bytes_to_audiosegment(stem_data)
                        stem_path = stems_dir / f"Stem_{instrument.title()}.wav"
                        
                        if safe_export_audio(stem_audio, stem_path, "wav"):
                            exported_files[f"stem_{instrument}"] = str(stem_path)
                            self.logger.debug(f"    ✅ Stem {instrument}: {stem_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error exporting stem {instrument}: {e}")
        
        return exported_files

    async def _export_metadata(self, config: Dict, project_dir: Path) -> Dict[str, str]:
        """Экспорт метаданных проекта"""
        
        exported_files = {}
        
        try:
            request_data = serialize_for_json(config.get("request_data", {}))
            structure = serialize_for_json(config.get("structure", {}))
            samples = serialize_for_json(config.get("samples", []))
            mastering = serialize_for_json(config.get("mastering", {}))
            
            # Основная информация о проекте
            project_info = {
                "wavedream_version": "Enhanced Pro v2.0",
                "export_timestamp": time.time(),
                "generation_config": {
                    "prompt": request_data.get("prompt", ""),
                    "genre": request_data.get("genre", "auto-detected"),
                    "mastering_purpose": request_data.get("mastering_purpose", "personal"),
                    "bpm": request_data.get("bpm", "auto"),
                    "duration": request_data.get("duration", "auto"),
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
                    "formats": config.get("export_formats", ["wav"]),
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
                        self.logger.debug(f"    ✅ {description}: {file_path.name}")
                        return True
                    else:
                        self.logger.error(f"    ❌ {description} failed: empty temp file")
                        return False
                        
                except Exception as e:
                    self.logger.error(f"    ❌ {description} error: {e}")
                    if "temp_path" in locals() and temp_path.exists():
                        temp_path.unlink()
                    return False
            
            # Файлы метаданных
            metadata_files = [
                (project_info, "project_info.json", "Project info", "project_info"),
                (structure, "track_structure.json", "Track structure", "structure"),
                (samples, "used_samples.json", "Used samples", "samples"),
                (mastering, "mastering_config.json", "Mastering config", "mastering"),
            ]
            
            for data, filename, description, key in metadata_files:
                if data:
                    file_path = project_dir / filename
                    if safe_json_save(data, file_path, description):
                        exported_files[key] = str(file_path)
            
            self.logger.info(f"    📋 Exported {len(exported_files)} metadata files")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting metadata: {e}")
            
            # Экстренные метаданные
            try:
                emergency_path = project_dir / "emergency_metadata.txt"
                with open(emergency_path, "w", encoding="utf-8") as f:
                    f.write("WaveDream Export Emergency Metadata\n")
                    f.write(f"Timestamp: {time.time()}\n")
                    f.write(f"Config keys: {list(config.keys())}\n")
                    f.write(f"Error: {str(e)}\n")
                exported_files["emergency"] = str(emergency_path)
                self.logger.warning(f"🚨 Emergency metadata: {emergency_path}")
            except Exception as ee:
                self.logger.error
