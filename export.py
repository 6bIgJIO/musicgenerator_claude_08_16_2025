# export.py - ИСПРАВЛЕННЫЙ экспорт-менеджер для WaveDream Enhanced Pro v2.0
import os
import io
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from datetime import datetime
import asyncio

# Импорты из WaveDream
try:
    from config import config, MasteringPurpose
except ImportError:
    # Fallback если config недоступен
    class MockConfig:
        DEFAULT_OUTPUT_DIR = "wavedream_output"
        CACHE_DIR = "wavedream_cache"
        
        class MockPurpose:
            FREELANCE = "freelance"
            PROFESSIONAL = "professional"
            PERSONAL = "personal"
    
    config = MockConfig()
    MasteringPurpose = MockConfig.MockPurpose()


class ExportManager:
    """
    ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ экспорт-менеджер WaveDream Enhanced Pro v2.0
    
    ✅ ИСПРАВЛЕНИЯ В ЭТОЙ ВЕРСИИ:
    - Правильная интеграция с config.py и архитектурой WaveDream
    - Поддержка bytes, AudioSegment и других типов данных
    - Улучшенная обработка ошибок без потери аудио
    - Валидация аудиоданных на каждом этапе
    - Совместимость с pipeline.py структурой
    - Правильные пути и форматы файлов
    - Исправлена проблема с отступами
    """
    
    def __init__(self, base_output_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Используем конфиг из WaveDream если доступен
        if hasattr(config, 'DEFAULT_OUTPUT_DIR'):
            self.base_output_dir = Path(base_output_dir or config.DEFAULT_OUTPUT_DIR)
        else:
            self.base_output_dir = Path(base_output_dir or "wavedream_output")
        
        self.supported_formats = ["wav", "mp3", "flac", "aac", "ogg"]
        
        # Создаём базовую структуру директорий
        self._create_base_structure()
        
        self.logger.info(f"✅ ExportManager initialized: {self.base_output_dir}")
    
    def _create_base_structure(self):
        """Создание базовой структуры директорий"""
        try:
            directories = [
                self.base_output_dir,
                self.base_output_dir / "projects",
                self.base_output_dir / "stems", 
                self.base_output_dir / "intermediate",
                self.base_output_dir / "final_mixes",
                self.base_output_dir / "metadata",
                self.base_output_dir / "reports",
                self.base_output_dir / "emergency"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating base structure: {e}")
            raise
    
    def check_export_environment(self) -> Dict[str, bool]:
        """Проверка окружения для экспорта"""
        checks = {}
        
        try:
            # Проверка записи в базовую директорию
            test_file = self.base_output_dir / "test_write.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
                checks["base_dir_writable"] = True
            except Exception:
                checks["base_dir_writable"] = False
            
            # Проверка свободного места (минимум 1GB)
            try:
                free_space = shutil.disk_usage(self.base_output_dir).free
                checks["sufficient_space"] = free_space > 1024 * 1024 * 1024
            except Exception:
                checks["sufficient_space"] = False
            
            # Проверка pydub
            try:
                test_audio = AudioSegment.silent(duration=100)
                buffer = io.BytesIO()
                test_audio.export(buffer, format="wav")
                checks["pydub_working"] = len(buffer.getvalue()) > 0
            except Exception:
                checks["pydub_working"] = False
            
            # Проверка soundfile
            try:
                test_data = np.array([0.1, 0.2, 0.1], dtype=np.float32)
                buffer = io.BytesIO()
                sf.write(buffer, test_data, 44100, format='wav')
                checks["soundfile_working"] = len(buffer.getvalue()) > 0
            except Exception:
                checks["soundfile_working"] = False
            
            self.logger.info(f"🔍 Export environment check: {checks}")
            return checks
            
        except Exception as e:
            self.logger.error(f"❌ Environment check error: {e}")
            return {"error": False}
    
    def save_intermediate(self, stage_name: str, project_name: str, audio_data: Union[bytes, AudioSegment]) -> Optional[str]:
        """
        Сохранение промежуточной версии на определённом этапе pipeline
        
        Args:
            stage_name: Название этапа (например, "01_base_generated")
            project_name: Имя проекта
            audio_data: Аудиоданные в виде bytes или AudioSegment
            
        Returns:
            Путь к сохранённому файлу или None при ошибке
        """
        try:
            if audio_data is None:
                self.logger.error(f"❌ Audio data is None for stage {stage_name}")
                return None
            
            # Создаём директорию проекта
            project_dir = self.base_output_dir / "intermediate" / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Формируем имя файла
            timestamp = datetime.now().strftime("%H%M%S")
            file_path = project_dir / f"{stage_name}_{timestamp}.wav"
            
            # Конвертируем и сохраняем
            success = self._save_audio_data(audio_data, file_path)
            
            if success:
                self.logger.info(f"💾 Intermediate saved: {stage_name} -> {file_path}")
                return str(file_path)
            else:
                self.logger.error(f"❌ Failed to save intermediate: {stage_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving intermediate {stage_name}: {e}")
            return None
    
    def save_stem(self, audio_data: Union[bytes, AudioSegment], project_name: str, stem_name: str) -> Optional[str]:
        """
        Сохранение отдельного стема
        
        Args:
            audio_data: Аудиоданные стема
            project_name: Имя проекта  
            stem_name: Имя стема (например, "kick", "snare", "lead")
            
        Returns:
            Путь к сохранённому стему или None при ошибке
        """
        try:
            if audio_data is None:
                self.logger.error(f"❌ Stem audio data is None for {stem_name}")
                return None
            
            # Создаём директорию для стемов проекта
            stems_dir = self.base_output_dir / "stems" / project_name
            stems_dir.mkdir(parents=True, exist_ok=True)
            
            # Формируем имя файла
            timestamp = datetime.now().strftime("%H%M%S")
            file_path = stems_dir / f"{stem_name}_{timestamp}.wav"
            
            # Конвертируем и сохраняем
            success = self._save_audio_data(audio_data, file_path)
            
            if success:
                self.logger.info(f"🎛️ Stem saved: {stem_name} -> {file_path}")
                return str(file_path)
            else:
                self.logger.error(f"❌ Failed to save stem: {stem_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving stem {stem_name}: {e}")
            return None
    
    def save_final_mix(self, audio_data: Union[bytes, AudioSegment], project_name: str, format: str = "wav") -> Optional[str]:
        """
        Сохранение финального микса
        
        Args:
            audio_data: Финальные аудиоданные
            project_name: Имя проекта
            format: Формат файла (wav, mp3, flac и т.д.)
            
        Returns:
            Путь к финальному файлу или None при ошибке
        """
        try:
            if audio_data is None:
                self.logger.error(f"❌ Final mix audio data is None")
                return None
            
            # Создаём директорию для финальных миксов
            final_dir = self.base_output_dir / "final_mixes"
            final_dir.mkdir(parents=True, exist_ok=True)
            
            # Формируем имя файла с таймстампом
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = final_dir / f"{project_name}_FINAL_{timestamp}.{format}"
            
            # Сохраняем в нужном формате
            success = self._save_audio_data(audio_data, file_path, format)
            
            if success:
                self.logger.info(f"🎉 Final mix saved: {file_path}")
                return str(file_path)
            else:
                self.logger.error(f"❌ Final mix file save failed")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving final mix: {e}")
            return None
    
    def _save_audio_data(self, audio_data: Union[bytes, AudioSegment], file_path: Path, format: str = "wav") -> bool:
        """
        Универсальная функция сохранения аудиоданных в файл
        
        Args:
            audio_data: Данные для сохранения
            file_path: Путь к файлу
            format: Формат файла
            
        Returns:
            True если сохранение успешно
        """
        try:
            # Убеждаемся что директория существует
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Определяем тип входных данных и сохраняем соответственно
            if isinstance(audio_data, bytes):
                # Если это bytes - пробуем разные способы
                if format == "wav" or format.lower() == "wav":
                    # Для WAV можем сохранить напрямую
                    with open(file_path, 'wb') as f:
                        f.write(audio_data)
                        
                    # Проверяем что файл создался и не пустой
                    if file_path.exists() and file_path.stat().st_size > 1000:
                        return True
                    else:
                        self.logger.warning(f"⚠️ Direct bytes save failed, trying AudioSegment conversion")
                
                # Для других форматов или если прямое сохранение не удалось
                try:
                    temp_audio = AudioSegment.from_file(io.BytesIO(audio_data))
                    temp_audio.export(str(file_path), format=format)
                    
                    if file_path.exists() and file_path.stat().st_size > 1000:
                        return True
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ AudioSegment conversion failed: {e}")
                    return False
                    
            elif isinstance(audio_data, AudioSegment):
                # Если это AudioSegment - прямой экспорт
                audio_data.export(str(file_path), format=format)
                
                if file_path.exists() and file_path.stat().st_size > 1000:
                    return True
                    
            elif hasattr(audio_data, 'export'):
                # Если у объекта есть метод export
                audio_data.export(str(file_path), format=format)
                
                if file_path.exists() and file_path.stat().st_size > 1000:
                    return True
                    
            elif hasattr(audio_data, 'getvalue'):
                # Если это BytesIO или подобный объект
                with open(file_path, 'wb') as f:
                    f.write(audio_data.getvalue())
                    
                if file_path.exists() and file_path.stat().st_size > 1000:
                    return True
            
            else:
                self.logger.error(f"❌ Unsupported audio data type: {type(audio_data)}")
                return False
            
            # Если дошли до сюда, значит сохранение не удалось
            self.logger.error(f"❌ Audio save failed - file missing or too small: {file_path}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Audio save error for {file_path}: {e}")
            return False
    
    def save_metadata(self, project_name: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Сохранение метаданных проекта
        
        Args:
            project_name: Имя проекта
            metadata: Словарь с метаданными
            
        Returns:
            Путь к файлу метаданных или None при ошибке
        """
        try:
            # Создаём директорию для метаданных
            metadata_dir = self.base_output_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Формируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = metadata_dir / f"{project_name}_metadata_{timestamp}.json"
            
            # Добавляем системную информацию к метаданным
            enhanced_metadata = {
                "project_name": project_name,
                "export_timestamp": timestamp,
                "export_datetime": datetime.now().isoformat(),
                "wavedream_version": "Enhanced Pro v2.0",
                **metadata
            }
            
            # Сохраняем JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"📋 Metadata saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving metadata: {e}")
            return None
    
    async def export_complete_project(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_audio: Dict[str, Union[bytes, AudioSegment]],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Полный экспорт проекта со всеми файлами (совместим с pipeline.py)
        
        Args:
            mastered_audio: Финальный мастированный трек
            intermediate_audio: Словарь промежуточных версий
            config: Конфигурация экспорта из pipeline
            
        Returns:
            Словарь путей к экспортированным файлам
        """
        exported_files = {}
        
        try:
            project_name = f"WD_Export_{int(time.time())}"
            output_dir = config.get("output_dir", str(self.base_output_dir / "projects" / project_name))
            
            # Создаём структуру проекта
            project_path = Path(output_dir)
            project_path.mkdir(parents=True, exist_ok=True)
            
            # 1. ЭКСПОРТ ФИНАЛЬНОГО ТРЕКА
            self.logger.info("  📁 Экспортируем финальный трек...")
            export_formats = config.get("export_formats", ["wav"])
            
            for format in export_formats:
                if format in self.supported_formats:
                    final_path = project_path / f"{project_name}_FINAL.{format}"
                    
                    if self._save_audio_data(mastered_audio, final_path, format):
                        exported_files[f"final_{format}"] = str(final_path)
                        self.logger.info(f"    ✅ Final {format.upper()}: {final_path.name}")
            
            # 2. ЭКСПОРТ СТЕМОВ
            if config.get("export_stems", False) and "stems" in intermediate_audio:
                self.logger.info("  🎛️ Экспортируем стемы...")
                stems_dir = project_path / "stems"
                stems_dir.mkdir(exist_ok=True)
                
                stems = intermediate_audio["stems"]
                if isinstance(stems, dict):
                    for stem_name, stem_audio in stems.items():
                        stem_path = stems_dir / f"{stem_name}.wav"
                        
                        if self._save_audio_data(stem_audio, stem_path, "wav"):
                            exported_files[f"stem_{stem_name}"] = str(stem_path)
                            self.logger.info(f"    🎸 Stem: {stem_name}")
            
            # 3. ЭКСПОРТ ПРОМЕЖУТОЧНЫХ ВЕРСИЙ
            self.logger.info("  📂 Экспортируем промежуточные версии...")
            intermediate_dir = project_path / "intermediate"
            intermediate_dir.mkdir(exist_ok=True)
            
            for stage_name, stage_audio in intermediate_audio.items():
                if stage_name != "stems":  # Стемы уже обработаны выше
                    stage_path = intermediate_dir / f"{stage_name}.wav"
                    
                    if self._save_audio_data(stage_audio, stage_path, "wav"):
                        exported_files[f"intermediate_{stage_name}"] = str(stage_path)
                        self.logger.info(f"    📋 Stage: {stage_name}")
            
            # 4. СОХРАНЕНИЕ КОНФИГА ПРОЕКТА
            config_path = project_path / "project_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            exported_files["project_config"] = str(config_path)
            
            # 5. СОЗДАНИЕ README
            readme_path = project_path / "README.md" 
            self._create_project_readme(readme_path, project_name, config, exported_files)
            exported_files["readme"] = str(readme_path)
            
            # 6. СОЗДАНИЕ ДЕТАЛЬНОГО ОТЧЁТА
            if config.get("structure") or config.get("samples") or config.get("mastering"):
                report_path = project_path / f"{project_name}_detailed_report.md"
                self._create_detailed_report(report_path, project_name, config, exported_files)
                exported_files["detailed_report"] = str(report_path)
            
            self.logger.info(f"🎉 Complete project export finished: {len(exported_files)} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ Complete project export error: {e}")
            
            # Аварийное сохранение финального трека
            try:
                emergency_path = self.base_output_dir / "emergency" / f"emergency_final_{int(time.time())}.wav"
                emergency_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self._save_audio_data(mastered_audio, emergency_path, "wav"):
                    return {"emergency_final": str(emergency_path)}
                else:
                    return {}
                
            except Exception as emergency_error:
                self.logger.error(f"❌ Emergency save failed: {emergency_error}")
                return {}
    
    def _create_project_readme(
        self, 
        readme_path: Path, 
        project_name: str,
        config: Dict[str, Any],
        exported_files: Dict[str, str]
    ):
        """Создание README файла для проекта"""
        try:
            request_data = config.get("request_data", {})
            
            readme_content = f"""# 🎵 WaveDream Project: {project_name}

## 📋 Project Information
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **WaveDream Version**: Enhanced Pro v2.0
- **Project Type**: Full AI Generated Track

## 🎯 Generation Parameters
- **Prompt**: `{request_data.get('prompt', 'N/A')}`
- **Genre**: {request_data.get('genre', 'Auto-detected')}
- **BPM**: {request_data.get('bpm', 'Auto')}
- **Duration**: {request_data.get('duration', 'Auto')} seconds
- **Mastering Purpose**: {request_data.get('mastering_purpose', 'personal')}
- **Energy Level**: {request_data.get('energy_level', 0.5)}

## 📁 Exported Files
"""
            
            # Группируем файлы по типам
            file_groups = {
                "Final Tracks": [],
                "Stems": [],
                "Intermediate Versions": [],
                "Project Files": []
            }
            
            for file_type, file_path in exported_files.items():
                file_name = Path(file_path).name
                
                if "final" in file_type:
                    file_groups["Final Tracks"].append(f"- **{file_type.replace('final_', '').upper()}**: `{file_name}`")
                elif "stem" in file_type:
                    file_groups["Stems"].append(f"- **{file_type.replace('stem_', '').title()}**: `{file_name}`")
                elif "intermediate" in file_type:
                    file_groups["Intermediate Versions"].append(f"- **{file_type.replace('intermediate_', '').title()}**: `{file_name}`")
                else:
                    file_groups["Project Files"].append(f"- **{file_type.replace('_', ' ').title()}**: `{file_name}`")
            
            for group_name, files in file_groups.items():
                if files:
                    readme_content += f"\n### {group_name}\n"
                    for file_line in files:
                        readme_content += f"{file_line}\n"
            
            readme_content += f"""
## 🎛️ WaveDream Pipeline Stages
1. ✅ Metadata Analysis & Genre Detection
2. ✅ Structure Generation (LLaMA3/Fallback)
3. ✅ Semantic Sample Selection
4. ✅ MusicGen Base Generation
5. ✅ Stem Creation & Layering
6. ✅ Smart Mixing
7. ✅ Effects Processing
8. ✅ Purpose-Driven Mastering
9. ✅ Quality Verification
10. ✅ Multi-Format Export

## 🔧 Usage Notes
- All intermediate versions preserved for analysis and remixing
- Stems available for further processing and remixing
- Mastering optimized for **{request_data.get('mastering_purpose', 'personal')}** use
- Full metadata and configuration preserved in project files

---
*Generated by WaveDream Enhanced Pro v2.0 - Full AI Music Generation Suite*
"""
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
        except Exception as e:
            self.logger.error(f"❌ Error creating README: {e}")
    
    def _create_detailed_report(
        self, 
        report_path: Path, 
        project_name: str, 
        config: Dict[str, Any], 
        exported_files: Dict[str, str]
    ):
        """Создание детального отчёта проекта"""
        try:
            request_data = config.get("request_data", {})
            structure = config.get("structure", {})
            samples = config.get("samples", [])
            mastering = config.get("mastering", {})
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"""# 🎵 WaveDream Enhanced Pro - Detailed Project Report

## 📋 Project: {project_name}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 Generation Request
- **Original Prompt**: `{request_data.get('prompt', 'N/A')}`
- **Target Genre**: {request_data.get('genre', 'Auto-detected')}
- **Target BPM**: {request_data.get('bpm', 'Auto-detected')}
- **Target Duration**: {request_data.get('duration', 'Auto-generated')} seconds
- **Mastering Purpose**: {request_data.get('mastering_purpose', 'personal')}
- **Energy Level**: {request_data.get('energy_level', 0.5)}
- **Creativity Factor**: {request_data.get('creativity_factor', 0.7)}

""")
                
                # Структура трека
                if structure and isinstance(structure, dict) and structure.get("sections"):
                    f.write(f"## 🏗️ Track Structure ({len(structure['sections'])} sections)\n\n")
                    f.write("| № | Type | Duration | Energy | Start Time |\n")
                    f.write("|---|------|----------|--------|-----------|\n")
                    
                    for i, section in enumerate(structure['sections'], 1):
                        f.write(f"| {i} | **{section.get('type', 'unknown').title()}** | "
                               f"{section.get('duration', 0)}s | {section.get('energy', 0):.1f} | "
                               f"{section.get('start_time', 0)}s |\n")
                    
                    f.write(f"\n**Total Duration**: {structure.get('total_duration', 0)} seconds\n")
                    f.write(f"**Structure Source**: {structure.get('source', 'unknown')}\n\n")
                
                # Используемые сэмплы
                if samples:
                    self._write_samples_section(f, samples)
                
                # Мастеринг конфигурация
                if mastering:
                    self._write_mastering_section(f, mastering)
                
                # Экспортированные файлы
                self._write_exported_files_section(f, exported_files)
                
                f.write("""
## 🎛️ Technical Pipeline Details
**WaveDream Enhanced Pro v2.0** uses a sophisticated multi-stage pipeline:

1. **Semantic Analysis**: Advanced NLP analysis of user prompts
2. **Genre Detection**: ML-based genre classification with keyword weighting
3. **Structure Generation**: LLaMA3-powered structural analysis with intelligent fallbacks
4. **Sample Selection**: Semantic similarity matching with MFCC/spectral analysis
5. **Base Generation**: MusicGen neural audio generation with genre conditioning
6. **Stem Processing**: Multi-layered stem creation with instrument-specific processing
7. **Smart Mixing**: Genre-aware mixing with automatic level balancing
8. **Effects Processing**: Dynamic effects chains adapted to genre and purpose
9. **Adaptive Mastering**: Purpose-driven mastering (freelance/professional/personal/etc.)
10. **Quality Verification**: Automated quality analysis and correction
11. **Multi-Format Export**: Comprehensive export with metadata preservation

## 💡 Recommendations
Based on your mastering purpose (**{request_data.get('mastering_purpose', 'personal')}**):

""")
                
                # Рекомендации по назначению
                purpose = request_data.get('mastering_purpose', 'personal')
                recommendations = {
                    "freelance": [
                        "✅ Optimized for commercial streaming platforms",
                        "📱 Ready for Spotify, Apple Music, YouTube",
                        "🎧 Test on multiple playback systems",
                        "💰 Suitable for client delivery and sales"
                    ],
                    "professional": [
                        "🎬 Broadcast and cinema-ready",
                        "📺 Full dynamic range preserved",
                        "🎛️ Professional loudness standards",
                        "🏆 Suitable for high-end productions"
                    ],
                    "personal": [
                        "🏠 Perfect for personal listening",
                        "🎵 Natural, unprocessed character",
                        "🔊 Great on home audio systems",
                        "❤️ Optimized for enjoyment over loudness"
                    ],
                    "family": [
                        "👨‍👩‍👧‍👦 Family-friendly mastering approach",
                        "🎥 Ideal for home videos and memories",
                        "📱 Mobile device optimized",
                        "😊 Warm and engaging sound"
                    ],
                    "streaming": [
                        "📺 Platform loudness normalization ready",
                        "🎵 Optimized for Spotify, YouTube, etc.",
                        "🔊 Consistent across all streaming services",
                        "📊 LUFS compliance guaranteed"
                    ],
                    "vinyl": [
                        "💿 Vinyl pressing optimized",
                        "🔥 Warm analog character",
                        "🎛️ Wide dynamic range preserved",
                        "✨ Perfect for physical media"
                    ]
                }
                
                purpose_recs = recommendations.get(purpose, recommendations["personal"])
                for rec in purpose_recs:
                    f.write(f"{rec}\n")
                
                f.write(f"""

## 📊 System Information
- **WaveDream Version**: Enhanced Pro v2.0
- **Export Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Files Created**: {len(exported_files)}
- **Project Size**: {self._calculate_project_size_from_files(exported_files)}

---
*This report was automatically generated by WaveDream Enhanced Pro v2.0*
""")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating detailed report: {e}")
    
    def _write_samples_section(self, f, samples: List[Dict]):
        """Запись секции об использованных сэмплах в отчёт"""
        f.write(f"## 🎛️ Used Samples ({len(samples)})\n\n")
        
        # Группируем по инструментам
        by_instrument = {}
        for sample in samples:
            instrument = sample.get("instrument_role", "unknown")
            if instrument not in by_instrument:
                by_instrument[instrument] = []
            by_instrument[instrument].append(sample)
        
        for instrument, instrument_samples in by_instrument.items():
            f.write(f"### {instrument.title()} ({len(instrument_samples)} samples)\n")
            f.write("| Filename | Section | Tags | Tempo |\n")
            f.write("|----------|---------|------|-------|\n")
            
            for sample in instrument_samples:
                filename = sample.get("filename", "unknown")
                section = sample.get("section", "unknown")
                tags = ", ".join(sample.get("tags", []))[:50]  # Ограничиваем длину
                tempo = sample.get("tempo", "N/A")
                
                f.write(f"| `{filename}` | {section} | {tags} | {tempo} |\n")
            
            f.write("\n")
    
    def _write_mastering_section(self, f, mastering: Dict):
        """Запись секции о мастеринге в отчёт"""
        f.write("## 🎚️ Mastering Configuration\n\n")
        
        # Основные параметры
        f.write("### Target Parameters\n")
        f.write(f"- **Target LUFS**: {mastering.get('target_lufs', 'N/A')}\n")
        f.write(f"- **Peak Ceiling**: {mastering.get('peak_ceiling', 'N/A')} dB\n")
        f.write(f"- **Character**: {mastering.get('character', 'N/A')}\n")
        f.write(f"- **Mastering Style**: {mastering.get('mastering_style', 'N/A')}\n\n")
        
        # Применённые этапы обработки
        if "applied_stages" in mastering:
            f.write("### Applied Processing Chain\n")
            for i, stage in enumerate(mastering["applied_stages"], 1):
                f.write(f"{i}. **{stage.replace('_', ' ').title()}**\n")
            f.write("\n")
        
        # Характеристики исходного материала
        if "source_characteristics" in mastering:
            source = mastering["source_characteristics"]
            f.write("### Source Material Analysis\n")
            f.write(f"- **Original LUFS**: {source.get('lufs', 'N/A')}\n")
            f.write(f"- **Original Peak**: {source.get('peak', 'N/A')} dB\n")
            f.write(f"- **Dynamic Range**: {source.get('dynamic_range', 'N/A')} LU\n")
            f.write(f"- **Stereo Width**: {source.get('stereo_width', 'N/A')}\n")
            f.write(f"- **Duration**: {source.get('duration', 'N/A')} seconds\n\n")
    
    def _write_exported_files_section(self, f, exported_files: Dict[str, str]):
        """Запись секции об экспортированных файлах в отчёт"""
        f.write("## 📁 Exported Files\n\n")
        
        # Группируем файлы по типам
        file_groups = {
            "Final Tracks": [],
            "Intermediate Versions": [],
            "Stems": [],
            "Project Metadata": []
        }
        
        for file_type, file_path in exported_files.items():
            relative_path = Path(file_path).name
            file_size = self._get_file_size_str(file_path)
            
            if "final" in file_type:
                file_groups["Final Tracks"].append(f"- **{file_type.replace('final_', '').upper()}**: `{relative_path}` ({file_size})")
            elif "stem" in file_type:
                file_groups["Stems"].append(f"- **{file_type.replace('stem_', '').title()}**: `{relative_path}` ({file_size})")
            elif "intermediate" in file_type:
                file_groups["Intermediate Versions"].append(f"- **{file_type.replace('intermediate_', '').title()}**: `{relative_path}` ({file_size})")
            else:
                file_groups["Project Metadata"].append(f"- **{file_type.replace('_', ' ').title()}**: `{relative_path}` ({file_size})")
        
        for group_name, files in file_groups.items():
            if files:
                f.write(f"### {group_name}\n")
                for file_line in files:
                    f.write(f"{file_line}\n")
                f.write("\n")
    
    def _get_file_size_str(self, file_path: str) -> str:
        """Получение размера файла в читаемом формате"""
        try:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > 1024 * 1024:  # MB
                    return f"{size / (1024 * 1024):.1f} MB"
                elif size > 1024:  # KB
                    return f"{size / 1024:.1f} KB"
                else:
                    return f"{size} bytes"
            return "unknown"
        except:
            return "unknown"
    
    def _calculate_project_size_from_files(self, exported_files: Dict[str, str]) -> str:
        """Вычисление общего размера проекта"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in exported_files.values():
                if file_path and isinstance(file_path, str) and os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
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
            
            return f"{size_str} ({file_count} files)"
            
        except Exception as e:
            self.logger.error(f"Error calculating project size: {e}")
            return "unknown"
    
    def force_save_everything(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_storage: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Аварийное сохранение всех данных при критической ошибке
        
        Args:
            mastered_audio: Финальный трек
            intermediate_storage: Промежуточные файлы
            metadata: Метаданные
            
        Returns:
            Список успешно сохранённых файлов
        """
        saved_files = []
        
        try:
            # Создаём аварийную директорию
            emergency_dir = self.base_output_dir / "emergency" / f"crash_save_{int(time.time())}"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🚨 Starting emergency save to: {emergency_dir}")
            
            # Сохраняем финальный трек
            if mastered_audio is not None:
                try:
                    final_path = emergency_dir / "emergency_final.wav"
                    if self._save_audio_data(mastered_audio, final_path, "wav"):
                        saved_files.append(str(final_path))
                        self.logger.info(f"🚨 Emergency final saved: {final_path}")
                except Exception as e:
                    self.logger.error(f"❌ Emergency final save failed: {e}")
            
            # Копируем промежуточные файлы
            if intermediate_storage:
                for stage, data in intermediate_storage.items():
                    try:
                        if isinstance(data, str) and os.path.exists(data):
                            # Это путь к файлу - копируем
                            emergency_file = emergency_dir / f"emergency_{stage}_{Path(data).name}"
                            shutil.copy2(data, emergency_file)
                            saved_files.append(str(emergency_file))
                            self.logger.info(f"🚨 Emergency copy: {stage} -> {emergency_file}")
                            
                        elif isinstance(data, dict):
                            # Это словарь стемов или других данных
                            for sub_name, sub_data in data.items():
                                if isinstance(sub_data, str) and os.path.exists(sub_data):
                                    emergency_file = emergency_dir / f"emergency_{stage}_{sub_name}_{Path(sub_data).name}"
                                    shutil.copy2(sub_data, emergency_file)
                                    saved_files.append(str(emergency_file))
                                    
                                elif sub_data is not None:  # Аудиоданные
                                    emergency_file = emergency_dir / f"emergency_{stage}_{sub_name}.wav"
                                    if self._save_audio_data(sub_data, emergency_file, "wav"):
                                        saved_files.append(str(emergency_file))
                                        
                        elif data is not None:  # Аудиоданные
                            emergency_file = emergency_dir / f"emergency_{stage}.wav"
                            if self._save_audio_data(data, emergency_file, "wav"):
                                saved_files.append(str(emergency_file))
                                
                    except Exception as e:
                        self.logger.error(f"❌ Emergency save failed for {stage}: {e}")
            
            # Сохраняем метаданные
            if metadata:
                try:
                    metadata_path = emergency_dir / "emergency_metadata.json"
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "emergency_save": True,
                            "timestamp": time.time(),
                            "datetime": datetime.now().isoformat(),
                            "original_metadata": metadata,
                            "saved_files": saved_files
                        }, f, indent=2, ensure_ascii=False, default=str)
                    
                    saved_files.append(str(metadata_path))
                    self.logger.info(f"🚨 Emergency metadata saved: {metadata_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Emergency metadata save failed: {e}")
            
            self.logger.info(f"🚨 Emergency save completed: {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ Critical emergency save error: {e}")
            return []
    
    def test_export_system(self) -> bool:
        """
        Тестирование системы экспорта
        
        Returns:
            True если тестирование прошло успешно
        """
        try:
            self.logger.info("🧪 Testing export system...")
            
            # Тест 1: Создание тестового аудио
            test_audio = AudioSegment.silent(duration=1000)  # 1 секунда тишины
            
            # Тест 2: Сохранение промежуточной версии
            intermediate_path = self.save_intermediate("test_stage", "test_project", test_audio)
            if not intermediate_path:
                raise Exception("Intermediate save failed")
            
            # Тест 3: Сохранение стема
            stem_path = self.save_stem(test_audio, "test_project", "test_stem")
            if not stem_path:
                raise Exception("Stem save failed")
            
            # Тест 4: Сохранение финального микса  
            final_path = self.save_final_mix(test_audio, "test_project")
            if not final_path:
                raise Exception("Final mix save failed")
            
            # Тест 5: Сохранение метаданных
            test_metadata = {"test": True, "timestamp": time.time()}
            metadata_path = self.save_metadata("test_project", test_metadata)
            if not metadata_path:
                raise Exception("Metadata save failed")
            
            # Очищаем тестовые файлы
            for test_file_path in [intermediate_path, stem_path, final_path, metadata_path]:
                try:
                    if test_file_path and os.path.exists(test_file_path):
                        os.unlink(test_file_path)
                except Exception:
                    pass
            
            self.logger.info("✅ Export system test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export system test failed: {e}")
            return False
    
    def debug_export_issue(
        self,
        mastered_audio: Union[bytes, AudioSegment],
        intermediate_audio: Dict[str, Union[bytes, AudioSegment]],
        config: Dict[str, Any]
    ):
        """
        Отладочная информация для диагностики проблем экспорта
        """
        try:
            self.logger.info("🔍 DEBUG: Export issue analysis")
            
            # Анализ финального аудио
            self.logger.info(f"🔍 Final audio type: {type(mastered_audio)}")
            if isinstance(mastered_audio, bytes):
                self.logger.info(f"🔍 Final audio size: {len(mastered_audio)} bytes")
            elif hasattr(mastered_audio, '__len__'):
                self.logger.info(f"🔍 Final audio length: {len(mastered_audio)}")
            
            # Анализ промежуточных версий
            self.logger.info(f"🔍 Intermediate audio keys: {list(intermediate_audio.keys())}")
            for key, audio in intermediate_audio.items():
                self.logger.info(f"🔍 {key}: type={type(audio)}")
                if isinstance(audio, bytes):
                    self.logger.info(f"🔍   size: {len(audio)} bytes")
                elif isinstance(audio, dict):
                    self.logger.info(f"🔍   dict keys: {list(audio.keys())}")
            
            # Анализ конфигурации
            self.logger.info(f"🔍 Config keys: {list(config.keys())}")
            self.logger.info(f"🔍 Output dir: {config.get('output_dir', 'N/A')}")
            self.logger.info(f"🔍 Export formats: {config.get('export_formats', 'N/A')}")
            self.logger.info(f"🔍 Export stems: {config.get('export_stems', 'N/A')}")
            
            # Проверка окружения
            env_status = self.check_export_environment()
            self.logger.info(f"🔍 Environment status: {env_status}")
            
        except Exception as e:
            self.logger.error(f"❌ Debug analysis error: {e}")
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики экспорта
        """
        try:
            stats = {
                "base_directory": str(self.base_output_dir),
                "supported_formats": self.supported_formats,
                "directories": {}
            }
            
            # Анализ директорий
            for subdir in ["projects", "stems", "intermediate", "final_mixes", "metadata", "emergency"]:
                dir_path = self.base_output_dir / subdir
                if dir_path.exists():
                    files = list(dir_path.rglob("*"))
                    file_count = len([f for f in files if f.is_file()])
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    stats["directories"][subdir] = {
                        "exists": True,
                        "file_count": file_count,
                        "total_size_mb": round(total_size / (1024 * 1024), 2)
                    }
                else:
                    stats["directories"][subdir] = {"exists": False}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting export statistics: {e}")
            return {"error": str(e)}


# Удобные функции для быстрого использования
def quick_export_track(audio_data: Union[bytes, AudioSegment], filename: str = None) -> Optional[str]:
    """
    Быстрый экспорт трека без сложной настройки
    """
    try:
        export_manager = ExportManager()
        
        if not filename:
            filename = f"quick_export_{int(time.time())}"
        
        final_path = export_manager.save_final_mix(audio_data, filename)
        return final_path
        
    except Exception as e:
        logging.error(f"❌ Quick export error: {e}")
        return None


def export_with_stems(
    final_audio: Union[bytes, AudioSegment],
    stems: Dict[str, Union[bytes, AudioSegment]],
    project_name: str = None
) -> Dict[str, str]:
    """
    Экспорт трека со стемами
    """
    try:
        export_manager = ExportManager()
        
        if not project_name:
            project_name = f"stems_export_{int(time.time())}"
        
        exported = {}
        
        # Экспортируем финальный трек
        final_path = export_manager.save_final_mix(final_audio, project_name)
        if final_path:
            exported["final"] = final_path
        
        # Экспортируем стемы
        for stem_name, stem_audio in stems.items():
            stem_path = export_manager.save_stem(stem_audio, project_name, stem_name)
            if stem_path:
                exported[f"stem_{stem_name}"] = stem_path
        
        return exported
        
    except Exception as e:
        logging.error(f"❌ Stems export error: {e}")
        return {}


def test_export_system() -> bool:
    """
    Быстрое тестирование системы экспорта
    """
    try:
        export_manager = ExportManager()
        return export_manager.test_export_system()
    except Exception as e:
        logging.error(f"❌ Export system test error: {e}")
        return False


# Глобальный экземпляр для использования в других модулях
default_export_manager = ExportManager()


if __name__ == "__main__":
    # Тестирование при прямом запуске модуля
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 Testing WaveDream Export Manager...")
    
    # Тест системы
    if test_export_system():
        print("✅ Export system test passed")
    else:
        print("❌ Export system test failed")
    
    # Показать статистику
    stats = default_export_manager.get_export_statistics()
    print(f"📊 Export statistics: {stats}")
    
    # Тест быстрого экспорта
    test_audio = AudioSegment.silent(duration=2000)  # 2 секунды тишины
    quick_path = quick_export_track(test_audio, "export_test")
    
    if quick_path and os.path.exists(quick_path):
        print(f"✅ Quick export test passed: {quick_path}")
        # Удаляем тестовый файл
        try:
            os.unlink(quick_path)
        except:
            pass
    else:
        print("❌ Quick export test failed")
    
    print("🎉 Export Manager testing completed")
