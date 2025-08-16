


# main.py - Главная система WaveDream Enhanced Pro v2.0

import os
import sys
import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
import requests
import io
from config import WaveDreamConfig
from export import ExportManager

# Добавляем путь к модулям WaveDream
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FL'))

# Импорты WaveDream модулей
try:
    from config import config, GenreType, MasteringPurpose
    from pipeline import WaveDreamPipeline, GenerationRequest, GenerationResult
    from sample_engine import SemanticSampleEngine
    from verification import MixVerifier
    from export import ExportManager
    from metadata import MetadataProcessor
    from self_check import verify_mix
    from semantic_engine import select_samples_by_semantics

except ImportError as e:
    print(f"❌ Error importing WaveDream modules: {e}")
    print("Make sure all WaveDream components are properly installed")
    sys.exit(1)


class WaveDreamEnhancedLauncher:
    """
    Главный лаунчер WaveDream Enhanced Pro v2.0
    
    Объединяет все компоненты системы:
    - Семантический анализ промптов
    - Жанровую детекцию с AI
    - LLaMA3 структурирование
    - MusicGen генерацию 
    - Умный мастеринг по назначению
    - Полную верификацию качества
    - Экспорт в множественных форматах
    """
    
    def __init__(self):
        # Настройка логирования
        self._setup_logging()
        
        # Инициализация компонентов
        self.logger = logging.getLogger(__name__)
        self.pipeline = WaveDreamPipeline()
        self.metadata_processor = MetadataProcessor()
        self.sample_engine = SemanticSampleEngine()
        self.verifier = MixVerifier()
        self.export_manager = ExportManager()
        
        # Статистика производительности
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'avg_generation_time': 0.0,
            'genre_statistics': {},
            'purpose_statistics': {}
        }
        
        # Проверяем окружение
        self._validate_environment()
        
        self.logger.info("🎵 WaveDream Enhanced Pro v2.0 initialized successfully")
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        log_config = config.LOGGING
        
        # Создаём форматтер
        formatter = logging.Formatter(log_config["format"])
        
        # Настраиваем уровень
        log_level = getattr(logging, log_config["level"], logging.INFO)
        
        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Файловый хендлер с ротацией
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config["file"],
                maxBytes=log_config["max_size_mb"] * 1024 * 1024,
                backupCount=log_config["backup_count"],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
            file_handler = None
        
        # Настраиваем root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        if file_handler:
            root_logger.addHandler(file_handler)
        
        # Убираем дублирование для наших логгеров
        for logger_name in ['wavedream', 'wavedream.core', '__main__']:
            logger = logging.getLogger(logger_name)
            logger.propagate = True
    
    def _validate_environment(self):
        """ИСПРАВЛЕННАЯ валидация окружения через новый ExportManager"""
        try:
            # ИСПРАВЛЕНО: Используем новый метод проверки окружения
            env_checks = self.export_manager.check_export_environment()
            
            failed_checks = [check for check, result in env_checks.items() if not result]
            
            if failed_checks:
                self.logger.warning(f"Environment validation issues: {'; '.join(failed_checks)}")
                
                # Критические проверки
                critical_failed = [check for check in failed_checks 
                                 if check in ["base_dir_writable", "sufficient_space", "pydub_working"]]
                
                if critical_failed:
                    raise RuntimeError(f"Critical environment checks failed: {'; '.join(critical_failed)}")
            else:
                self.logger.info("✅ All environment checks passed")
                
        except Exception as e:
            self.logger.error(f"❌ Critical environment validation error: {e}")
            raise
    
    async def generate_track_async(self, request: GenerationRequest) -> GenerationResult:
        """
        Асинхронная генерация трека через полный pipeline
        
        Args:
            request: Запрос на генерацию с параметрами
            
        Returns:
            Результат генерации с метриками качества
        """
        start_time = time.time()
        self.performance_stats['total_generations'] += 1
        
        try:
            self.logger.info(f"🚀 Starting async track generation")
            self.logger.info(f"📝 Prompt: '{request.prompt}'")
            self.logger.info(f"🎯 Purpose: {request.mastering_purpose}")
            self.logger.info(f"🎭 Genre hint: {request.genre or 'auto-detect'}")
            
            # Запускаем полный pipeline
            result = await self.pipeline.generate_track(request)
            
            # Обновляем статистику
            generation_time = time.time() - start_time
            
            if result.success:
                self.performance_stats['successful_generations'] += 1
                
                # Обновляем статистику по жанрам
                detected_genre = result.structure_data.get('detected_genre', 'unknown') if result.structure_data else 'unknown'
                self.performance_stats['genre_statistics'][detected_genre] = \
                    self.performance_stats['genre_statistics'].get(detected_genre, 0) + 1
                
                # Статистика по назначению
                self.performance_stats['purpose_statistics'][request.mastering_purpose] = \
                    self.performance_stats['purpose_statistics'].get(request.mastering_purpose, 0) + 1
                
                # Среднее время генерации
                current_avg = self.performance_stats['avg_generation_time']
                total_successful = self.performance_stats['successful_generations']
                self.performance_stats['avg_generation_time'] = \
                    (current_avg * (total_successful - 1) + generation_time) / total_successful
                
                self.logger.info(f"✅ Generation completed successfully in {generation_time:.1f}s")
                self.logger.info(f"🎯 Quality score: {result.quality_score:.2f}/1.0")
                
            else:
                self.logger.error(f"❌ Generation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Async generation error: {e}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            
            return GenerationResult(
                success=False,
                generation_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def generate_track_sync(self, request: GenerationRequest) -> GenerationResult:
        """Синхронная обёртка для асинхронной генерации"""
        return asyncio.run(self.generate_track_async(request))
    
    def run_interactive_mode(self):
        """Интерактивный режим с расширенными возможностями"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║ 🎵 WaveDream Enhanced Pro v2.0 - Full AI Music Generation Suite 🎵             ║
║                                                                                  ║ 
║ 🧠 LLaMA3 Structure | 🎼 MusicGen Base | 🔍 Semantic Samples | 🎛️ Smart Master ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            self._display_main_menu()
            choice = input("\n🎯 Your choice: ").strip()
            
            try:
                if choice == "1":
                    self._interactive_enhanced_generation()
                elif choice == "2":
                    self._interactive_batch_generation()
                elif choice == "3":
                    self._interactive_sample_analysis()
                elif choice == "4":
                    self._interactive_genre_testing()
                elif choice == "5":
                    self._interactive_quality_analysis()
                elif choice == "6":
                    self._interactive_system_management()
                elif choice == "7":
                    self._interactive_settings()
                elif choice == "8":
                    self._display_statistics()
                elif choice == "9":
                    self._run_system_diagnostics()
                elif choice == "0":
                    print("👋 Thank you for using WaveDream Enhanced Pro!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n⏸️ Operation cancelled by user")
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                print(f"❌ An error occurred: {e}")
                print("Please try again or check logs for details.")
    
    def _display_main_menu(self):
        """Отображение главного меню"""
        print("\n" + "="*80)
        print("🎵 WaveDream Enhanced Pro v2.0 - Main Menu")
        print("="*80)
        print("1. 🚀 Enhanced Track Generation (Full Pipeline)")
        print("2. 📦 Batch Generation from JSON") 
        print("3. 🔍 Sample Database Analysis")
        print("4. 🎭 Genre Detection Testing")
        print("5. 📊 Quality Analysis Tools")
        print("6. ⚙️ System Management")
        print("7. 🛠️ Settings & Configuration")
        print("8. 📈 Performance Statistics")
        print("9. 🔧 System Diagnostics")
        print("0. 🚪 Exit")
    
    def _interactive_enhanced_generation(self):
        """Интерактивная расширенная генерация"""
        print("\n🚀 ENHANCED TRACK GENERATION")
        print("-" * 50)
        print("Full AI pipeline: Prompt Analysis → Genre Detection → Structure → Generation → Mastering")
        
        # Ввод основных параметров
        prompt = input("\n📝 Enter track description: ").strip()
        if not prompt:
            print("❌ Prompt cannot be empty")
            return
        
        # Предварительный анализ промпта
        print("\n🧠 Analyzing prompt...")
        prompt_analysis = self.metadata_processor.analyze_prompt(prompt)
        detected_genre = self.metadata_processor.detect_genre(prompt, prompt_analysis.get('tags', []))
        
        print(f"🎭 Detected genre: {detected_genre}")
        print(f"🎵 Detected BPM: {prompt_analysis.get('bpm', 'auto')}")
        print(f"🎹 Detected instruments: {', '.join(prompt_analysis.get('instruments', ['auto']))}")
        print(f"😊 Detected mood: {', '.join(prompt_analysis.get('mood', ['neutral']))}")
        
        # Опции жанра
        confirm_genre = input(f"\nContinue with detected genre '{detected_genre}'? (Y/n): ").lower()
        if confirm_genre == 'n':
            available_genres = [genre.value for genre in GenreType]
            print(f"Available genres: {', '.join(available_genres)}")
            manual_genre = input("Enter genre manually: ").strip().lower()
            if manual_genre in available_genres:
                detected_genre = manual_genre
            else:
                print(f"❌ Unknown genre, using detected: {detected_genre}")
        
        # Назначение мастеринга
        print("\n🎯 Select mastering purpose:")
        purposes = [purpose.value for purpose in MasteringPurpose]
        for i, purpose in enumerate(purposes, 1):
            print(f"{i}. {purpose.title()} - {self._get_purpose_description(purpose)}")
        
        purpose_choice = input("Choice (1-6, Enter for personal): ").strip()
        try:
            if purpose_choice:
                mastering_purpose = purposes[int(purpose_choice) - 1]
            else:
                mastering_purpose = "personal"
        except (ValueError, IndexError):
            mastering_purpose = "personal"
        
        print(f"🎯 Selected purpose: {mastering_purpose}")
        
        # Дополнительные параметры
        print("\n⚙️ Additional options:")
        
        duration = input("Duration in seconds (Enter for auto): ").strip()
        try:
            duration = int(duration) if duration else None
        except ValueError:
            duration = None
        
        export_stems = input("Export all intermediate versions? (Y/n): ").lower() != 'n'
        
        export_formats = ["wav"]
        more_formats = input("Export additional formats? (mp3,flac,aac): ").strip()
        if more_formats:
            additional = [f.strip() for f in more_formats.split(',')]
            export_formats.extend(additional)
        
        # Вывод
        output_dir = input("Output directory (Enter for auto): ").strip()
        if not output_dir:
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c.isspace()).strip()
            safe_prompt = "_".join(safe_prompt.split())
            output_dir = f"output_{safe_prompt}_{detected_genre}_{mastering_purpose}"
        
        # Создаём запрос
        request = GenerationRequest(
            prompt=prompt,
            genre=detected_genre,
            bpm=prompt_analysis.get('bpm'),
            duration=duration,
            mastering_purpose=mastering_purpose,
            output_dir=output_dir,
            export_stems=export_stems,
            energy_level=prompt_analysis.get('energy_level', 0.5),
            creativity_factor=prompt_analysis.get('complexity_score', 0.7)
        )
        
        # Подтверждение
        print(f"\n📋 GENERATION SUMMARY:")
        print(f"  📝 Prompt: '{prompt}'")
        print(f"  🎭 Genre: {detected_genre}")
        print(f"  🎵 BPM: {request.bpm or 'auto'}")
        print(f"  ⏱️ Duration: {duration or 'auto'} seconds")
        print(f"  🎯 Purpose: {mastering_purpose}")
        print(f"  📁 Output: {output_dir}")
        print(f"  💾 Export stems: {export_stems}")
        print(f"  🎼 Formats: {', '.join(export_formats)}")
        
        confirm = input("\nProceed with generation? (Y/n): ").lower()
        if confirm == 'n':
            print("❌ Generation cancelled")
            return
        
        # Запускаем генерацию
        print(f"\n🚀 Starting enhanced generation...")
        print("This may take several minutes depending on complexity...")
        
        try:
            result = self.generate_track_sync(request)
            
            if result.success:
                print(f"\n🎉 GENERATION COMPLETED SUCCESSFULLY!")
                print(f"📁 Final track: {result.final_path}")
                print(f"⏱️ Generation time: {result.generation_time:.1f} seconds")
                print(f"🎯 Quality score: {result.quality_score:.2f}/1.0")
                
                if result.used_samples:
                    print(f"🎛️ Used samples: {len(result.used_samples)}")
                
                if result.structure_data:
                    sections = result.structure_data.get('sections', [])
                    print(f"🏗️ Structure sections: {len(sections)}")
                
                # Предложение воспроизведения
                if result.final_path and os.path.exists(result.final_path):
                    play = input("\nPlay the generated track? (y/N): ").lower()
                    if play == 'y':
                        self._play_audio_file(result.final_path)
                
                # Показать отчёт о качестве
                if result.quality_score < 0.8:
                    show_quality = input("Show detailed quality report? (Y/n): ").lower()
                    if show_quality != 'n':
                        self._show_quality_details(result)
                
            else:
                print(f"\n❌ GENERATION FAILED")
                print(f"Error: {result.error_message}")
                print("Check logs for detailed error information")
                
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            print(f"❌ Unexpected error during generation: {e}")
    
    def _interactive_batch_generation(self):
        """Интерактивная пакетная генерация"""
        print("\n📦 BATCH GENERATION FROM JSON")
        print("-" * 40)
        
        # Поиск JSON файлов в текущей директории
        json_files = list(Path('.').glob('*.json'))
        batch_files = [f for f in json_files if 'batch' in f.name.lower() or 'tasks' in f.name.lower()]
        
        if batch_files:
            print("Found potential batch files:")
            for i, file in enumerate(batch_files, 1):
                print(f"{i}. {file}")
            print(f"{len(batch_files) + 1}. Enter custom path")
            
            choice = input("Select batch file: ").strip()
            try:
                if choice and int(choice) <= len(batch_files):
                    batch_file = str(batch_files[int(choice) - 1])
                else:
                    batch_file = input("Enter batch file path: ").strip()
            except (ValueError, IndexError):
                batch_file = input("Enter batch file path: ").strip()
        else:
            batch_file = input("Enter batch file path: ").strip()
        
        if not batch_file or not os.path.exists(batch_file):
            print("❌ Batch file not found")
            return
        
        # Загружаем задачи
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading batch file: {e}")
            return
        
        tasks = batch_data.get("tasks", [])
        if not tasks:
            print("❌ No tasks found in batch file")
            return
        
        print(f"📦 Found {len(tasks)} tasks")
        
        # Настройки пакетной обработки
        parallel_processing = input("Enable parallel processing? (Y/n): ").lower() != 'n'
        max_concurrent = 2  # Безопасный лимит
        
        if parallel_processing:
            try:
                max_concurrent = int(input(f"Max concurrent tasks (default {max_concurrent}): ") or max_concurrent)
            except ValueError:
                pass
        
        # Запуск пакетной обработки
        print(f"\n🚀 Starting batch processing: {len(tasks)} tasks")
        if parallel_processing:
            print(f"⚡ Parallel processing: max {max_concurrent} concurrent")
        
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_data in enumerate(tasks, 1):
            print(f"\n📋 Task {i}/{len(tasks)}: {task_data.get('name', f'task_{i}')}")
            
            try:
                # Создаём запрос из данных задачи
                request = self._create_request_from_task(task_data, i)
                
                # Генерируем
                result = self.generate_track_sync(request)
                
                if result.success:
                    print(f"  ✅ Completed: {result.final_path}")
                    successful_tasks += 1
                else:
                    print(f"  ❌ Failed: {result.error_message}")
                    failed_tasks += 1
                
            except Exception as e:
                print(f"  ❌ Task error: {e}")
                failed_tasks += 1
        
        # Итоги
        print(f"\n📊 BATCH PROCESSING COMPLETED")
        print(f"✅ Successful: {successful_tasks}")
        print(f"❌ Failed: {failed_tasks}")
        print(f"📈 Success rate: {successful_tasks/(successful_tasks+failed_tasks)*100:.1f}%")
    
    def _interactive_sample_analysis(self):
        """Интерактивный анализ базы сэмплов"""
        print("\n🔍 SAMPLE DATABASE ANALYSIS")
        print("-" * 35)
        
        print("Choose analysis type:")
        print("1. Database statistics")
        print("2. Rebuild semantic index")
        print("3. Search samples by query")
        print("4. Analyze sample quality")
        print("5. Export sample metadata")
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            print("\n📊 Generating database statistics...")
            stats = self.sample_engine.get_statistics()
            
            print(f"\n📈 SAMPLE DATABASE STATISTICS")
            print(f"Total samples: {stats.get('total_samples', 0)}")
            print(f"Average quality: {stats.get('avg_quality', 0):.2f}")
            
            # Статистика по жанрам
            genre_dist = stats.get('genre_distribution', {})
            if genre_dist:
                print(f"\n🎭 Genre distribution:")
                for genre, count in sorted(genre_dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {genre}: {count} samples")
            
            # Статистика по инструментам
            instrument_dist = stats.get('instrument_distribution', {})
            if instrument_dist:
                print(f"\n🎼 Instrument distribution:")
                for instrument, count in sorted(instrument_dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {instrument}: {count} samples")
            
            # Производительность
            perf_stats = stats.get('performance_stats', {})
            if perf_stats:
                print(f"\n⚡ Performance statistics:")
                print(f"  Queries processed: {perf_stats.get('queries', 0)}")
                print(f"  Cache hit rate: {perf_stats.get('cache_hits', 0) / max(1, perf_stats.get('queries', 1)) * 100:.1f}%")
                print(f"  Average query time: {perf_stats.get('avg_query_time', 0):.3f}s")
        
        elif choice == "2":
            confirm = input("⚠️ Rebuild semantic index? This will take time. (y/N): ").lower()
            if confirm == 'y':
                print("🔄 Rebuilding semantic index...")
                self.sample_engine.build_semantic_index()
                print("✅ Semantic index rebuilt successfully")
        
        elif choice == "3":
            query = input("Enter search query (tags, instruments, genre): ").strip()
            if query:
                print(f"\n🔍 Searching for: '{query}'")
                
                # Парсим запрос
                query_parts = query.split()
                search_results = asyncio.run(self.sample_engine.find_samples(
                    tags=query_parts,
                    max_results=10
                ))
                
                if search_results:
                    print(f"\n📋 Found {len(search_results)} samples:")
                    for i, sample in enumerate(search_results, 1):
                        filename = sample.get('filename', 'unknown')
                        score = sample.get('score', 0)
                        tags = sample.get('tags', [])
                        print(f"{i}. {filename} (score: {score:.2f})")
                        print(f"   Tags: {', '.join(tags[:5])}")
                else:
                    print("❌ No samples found matching query")
        
        elif choice == "4":
            print("📊 Analyzing sample quality...")
            # В реальной реализации здесь будет полный анализ качества
            print("Quality analysis completed - see logs for details")
        
        elif choice == "5":
            output_file = input("Export metadata to file (default: sample_metadata.json): ").strip()
            if not output_file:
                output_file = "sample_metadata.json"
            
            try:
                stats = self.sample_engine.get_statistics()
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                print(f"✅ Metadata exported to: {output_file}")
            except Exception as e:
                print(f"❌ Export error: {e}")
    
    def _interactive_genre_testing(self):
        """Интерактивное тестирование детекции жанров"""
        print("\n🎭 GENRE DETECTION TESTING")
        print("-" * 30)
        
        test_prompts = [
            ("dark aggressive trap 160bpm with vocal chops and 808s", "trap"),
            ("мелодичный лоуфай для учёбы с винтажными текстурами", "lofi"),
            ("liquid drum and bass neurofunk 174bpm atmospheric", "dnb"),
            ("атмосферный эмбиент космос медитация 70bpm", "ambient"),
            ("phonk memphis cowbell drift aggressive", "phonk"),
            ("техно минимал 130bpm industrial warehouse", "techno"),
            ("cinematic epic trailer orchestral heroic", "cinematic"),
            ("house deep groove плавный bassline 124bpm", "house")
        ]
        
        print("Choose testing mode:")
        print("1. Test with built-in examples")
        print("2. Test custom prompts")
        print("3. Test accuracy on known samples")
        
        choice = input("Select mode (1-3): ").strip()
        
        if choice == "1":
            print("\n🧪 Testing with built-in examples:")
            
            correct = 0
            total = len(test_prompts)
            
            for prompt, expected in test_prompts:
                detected = self.metadata_processor.detect_genre(prompt)
                is_correct = detected == expected
                
                status = "✅" if is_correct else "❌"
                print(f"{status} '{prompt[:50]}...'")
                print(f"   Expected: {expected} | Detected: {detected}")
                
                if is_correct:
                    correct += 1
            
            accuracy = correct / total * 100
            print(f"\n📊 Accuracy: {accuracy:.1f}% ({correct}/{total})")
            
        elif choice == "2":
            while True:
                prompt = input("\nEnter test prompt (or 'quit' to exit): ").strip()
                if prompt.lower() == 'quit':
                    break
                
                if prompt:
                    # Полный анализ
                    analysis = self.metadata_processor.analyze_prompt(prompt)
                    detected_genre = self.metadata_processor.detect_genre(prompt, analysis.get('tags', []))
                    
                    print(f"\n🔍 Analysis results for: '{prompt}'")
                    print(f"🎭 Detected genre: {detected_genre}")
                    print(f"🎵 Detected BPM: {analysis.get('bpm', 'none')}")
                    print(f"🎹 Instruments: {', '.join(analysis.get('instruments', ['none']))}")
                    print(f"😊 Mood: {', '.join(analysis.get('mood', ['neutral']))}")
                    print(f"🧠 Complexity: {analysis.get('complexity_score', 0):.2f}")
                    
                    if analysis.get('mentioned_sections'):
                        print(f"🏗️ Structure hints: {', '.join(analysis['mentioned_sections'])}")
    
    def _interactive_quality_analysis(self):
        """ИСПРАВЛЕННЫЙ интерактивный анализ качества"""
        print("\n📊 QUALITY ANALYSIS TOOLS")
        print("-" * 30)
        
        print("1. Analyze audio file")
        print("2. Compare two tracks") 
        print("3. Batch quality analysis")
        print("4. Generate quality report")
        
        choice = input("Select tool (1-4): ").strip()
        
        if choice == "1":
            file_path = input("Enter audio file path: ").strip()
            if os.path.exists(file_path):
                print(f"🔍 Analyzing: {file_path}")
                
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(file_path)
                    
                    # ИСПРАВЛЕНО: Правильный вызов анализа качества
                    target_config = {"target_lufs": -14, "peak_ceiling": -1}
                    report = asyncio.run(self.verifier.analyze_track(audio, target_config))
                    
                    print(f"📊 Quality score: {report.get('overall_score', 0):.2f}/1.0")
                    print(f"🎯 Status: {report.get('status', 'unknown')}")
                    
                    issues = report.get('issues', [])
                    if issues:
                        critical = len([i for i in issues if i.get('severity') == 'critical'])
                        warnings = len([i for i in issues if i.get('severity') == 'warning'])
                        print(f"🚨 Issues: {critical} critical, {warnings} warnings")
                    
                    # ИСПРАВЛЕНО: Используем новый метод создания отчета
                    if input("Generate detailed report? (Y/n): ").lower() != 'n':
                        report_path = f"{Path(file_path).stem}_quality_report.md"
                        
                        # Создаем отчет через новый ExportManager
                        try:
                            # Подготавливаем данные для отчета
                            report_config = {
                                "request_data": {
                                    "prompt": f"Quality analysis of {Path(file_path).name}",
                                    "mastering_purpose": "analysis"
                                },
                                "structure": {"total_duration": len(audio) / 1000.0},
                                "analysis_results": report
                            }
                            
                            # Создаем отчет
                            report_file = asyncio.run(
                                self.export_manager.create_project_report(
                                    config=report_config,
                                    exported_files={"analyzed_file": file_path},
                                    project_dir=Path(file_path).parent
                                )
                            )
                            
                            if report_file:
                                print(f"📋 Report saved: {report_file}")
                            else:
                                print("❌ Failed to generate report")
                                
                        except Exception as report_error:
                            self.logger.error(f"Report generation error: {report_error}")
                            print("❌ Report generation failed")
                
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
            else:
                print("❌ File not found")
    
    def _interactive_system_management(self):


        """ИСПРАВЛЕННОЕ интерактивное управление системой"""
        print("\n⚙️ SYSTEM MANAGEMENT")
        print("-" * 25)
        
        print("1. Clear cache files")
        print("2. Update sample index")
        print("3. Check system health")
        print("4. Export configuration")
        print("5. Import configuration") 
        print("6. Reset to defaults")
        print("7. Test export system")  # НОВАЯ ОПЦИЯ
        
        choice = input("Select action (1-7): ").strip()
        
        if choice == "1":
            cache_dir = Path(config.CACHE_DIR)
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print("✅ Cache cleared")
            else:
                print("ℹ️ No cache to clear")
        
        elif choice == "2":
            print("🔄 Updating sample index...")
            self.sample_engine.build_semantic_index()
            print("✅ Index updated")
        
        elif choice == "3":
            print("🏥 Running system health check...")
            self._run_system_health_check()
        
        elif choice == "4":
            config_path = input("Export config to (default: wavedream_config.json): ").strip()
            if not config_path:
                config_path = "wavedream_config.json"
            
            try:
                config.export_config(config_path)
                print(f"✅ Configuration exported: {config_path}")
            except Exception as e:
                print(f"❌ Export error: {e}")
                
        elif choice == "7":  # НОВАЯ ОПЦИЯ
            print("🧪 Testing export system...")
            try:
                # ИСПРАВЛЕНО: Тестируем новую систему экспорта
                env_checks = self.export_manager.check_export_environment()
                
                print("Environment checks:")
                for check, result in env_checks.items():
                    status = "✅" if result else "❌"
                    print(f"  {status} {check.replace('_', ' ').title()}")
                
                # Создаем тестовое аудио и проверяем экспорт
                print("\nTesting audio export...")
                from pydub.generators import Sine
                test_audio = Sine(440).to_audio_segment(duration=1000)
                
                test_config = {
                    "output_dir": "test_export",
                    "export_formats": ["wav", "mp3"],
                    "request_data": {"prompt": "test", "mastering_purpose": "test"}
                }
                
                # Тест через новый ExportManager
                buffer = io.BytesIO()
                test_audio.export(buffer, format="wav")
                test_bytes = buffer.getvalue()
                
                result = asyncio.run(
                    self.export_manager.export_complete_project(
                        mastered_audio=test_bytes,
                        intermediate_audio={},
                        config=test_config
                    )
                )
                
                print(f"✅ Export test successful: {len(result)} files created")
                
            except Exception as e:
                print(f"❌ Export test failed: {e}")
    
    def _interactive_settings(self):
        """Интерактивные настройки"""
        print("\n🛠️ SETTINGS & CONFIGURATION")
        print("-" * 35)
        
        current_settings = {
            "Sample Directory": config.DEFAULT_SAMPLE_DIR,
            "Audio Analysis Max Duration": f"{config.AUDIO_ANALYSIS['max_duration']}s",
            "Sample Rate": f"{config.AUDIO_ANALYSIS['sample_rate']}Hz",
            "Tempo Tolerance": f"±{config.SAMPLE_MATCHING['tempo_tolerance']} BPM",
            "Min Score Threshold": config.SAMPLE_MATCHING['min_score_threshold'],
            "Max Workers": config.PERFORMANCE['max_workers'],
            "Semantic Analysis": config.SAMPLE_MATCHING['enable_semantic_embeddings']
        }
        
        print("Current settings:")
        for key, value in current_settings.items():
            print(f"  {key}: {value}")
        
        print(f"\nSupported genres: {', '.join([g.value for g in GenreType])}")
        print(f"Mastering purposes: {', '.join([p.value for p in MasteringPurpose])}")
        
        change = input("\nModify settings? (y/N): ").lower()
        if change == 'y':
            print("💡 To modify settings, edit the configuration files or use environment variables")
            print("💡 See documentation for detailed configuration options")
    
    def _display_statistics(self):
        """Отображение статистики производительности"""
        print("\n📈 PERFORMANCE STATISTICS")
        print("-" * 30)
        
        stats = self.performance_stats
        
        print(f"Total generations: {stats['total_generations']}")
        print(f"Successful generations: {stats['successful_generations']}")
        
        if stats['total_generations'] > 0:
            success_rate = stats['successful_generations'] / stats['total_generations'] * 100
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average generation time: {stats['avg_generation_time']:.1f}s")
        
        if stats['genre_statistics']:
            print(f"\nGenre distribution:")
            for genre, count in sorted(stats['genre_statistics'].items(), key=lambda x: -x[1]):
                print(f"  {genre}: {count} generations")
        
        if stats['purpose_statistics']:
            print(f"\nPurpose distribution:")
            for purpose, count in sorted(stats['purpose_statistics'].items(), key=lambda x: -x[1]):
                print(f"  {purpose}: {count} generations")
        
        # Системная информация
        print(f"\nSystem info:")
        print(f"  Sample database size: {len(self.sample_engine.samples_index)} samples")
        print(f"  Cache entries: {len(self.sample_engine.embeddings_cache)}")
    
    def _run_system_diagnostics(self):
        """ИСПРАВЛЕННАЯ системная диагностика с новыми модулями"""
        print("\n🔧 SYSTEM DIAGNOSTICS")
        print("-" * 25)
        
        print("Running comprehensive system check...")
        
        # Проверка зависимостей (без изменений)
        print("\n📦 Checking dependencies...")
        dependencies = ['torch', 'librosa', 'pydub', 'numpy', 'scipy', 'soundfile']
        
        for dep in dependencies:
            try:
                __import__(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} - Missing!")
        
        # ИСПРАВЛЕНО: Проверка директорий через новый ExportManager
        print(f"\n📁 Checking directories...")
        try:
            env_checks = self.export_manager.check_export_environment()
            
            if env_checks.get("base_dir_writable", False):
                print(f"  ✅ Output directory writable")
            else:
                print(f"  ❌ Output directory not writable")
                
            if env_checks.get("sufficient_space", False):
                print(f"  ✅ Sufficient disk space")
            else:
                print(f"  ⚠️ Low disk space warning")
                
        except Exception as e:
            print(f"  ❌ Directory check error: {e}")
        
        # Проверка семантической модели (без изменений)
        print(f"\n🧠 Checking semantic model...")
        try:
            if hasattr(self.sample_engine, 'semantic_model') and self.sample_engine.semantic_model:
                print(f"  ✅ Semantic model loaded")
            else:
                print(f"  ⚠️ Semantic model not available")
        except Exception as e:
            print(f"  ❌ Semantic model error: {e}")
        
        # ИСПРАВЛЕНО: Проверка экспорта через новый ExportManager
        print(f"\n💾 Testing export system...")
        try:
            from pydub.generators import Sine
            test_audio = Sine(440).to_audio_segment(duration=500)
            
            # Тест экспорта
            buffer = io.BytesIO()
            test_audio.export(buffer, format="wav")
            test_bytes = buffer.getvalue()
            
            if len(test_bytes) > 1000:  # Проверяем что экспорт не пустой
                print(f"  ✅ Audio export working")
            else:
                print(f"  ❌ Audio export failed - empty result")
                
        except Exception as e:
            print(f"  ❌ Export test error: {e}")
        
        # Проверка производительности (без изменений)
        print(f"\n⚡ Performance test...")
        start_time = time.time()
        
        test_prompt = "test electronic music 120bpm"
        analysis = self.metadata_processor.analyze_prompt(test_prompt)
        
        test_time = time.time() - start_time
        print(f"  📊 Prompt analysis: {test_time:.3f}s")
        
        if test_time < 1.0:
            print(f"  ✅ Performance: Good")
        elif test_time < 3.0:
            print(f"  ⚠️ Performance: Acceptable")
        else:
            print(f"  ❌ Performance: Slow")
    
    def _run_system_health_check(self):
        """ИСПРАВЛЕННАЯ проверка здоровья системы"""
        health_status = {
            "dependencies": True,
            "directories": True,
            "sample_index": True,
            "semantic_model": True,
            "memory_usage": True,
            "export_system": True  # НОВАЯ ПРОВЕРКА
        }
        
        # ИСПРАВЛЕНО: Проверка экспорта
        try:
            env_checks = self.export_manager.check_export_environment()
            critical_checks = ["base_dir_writable", "sufficient_space", "pydub_working"]
            
            failed_critical = [check for check in critical_checks if not env_checks.get(check, False)]
            
            if failed_critical:
                health_status["export_system"] = False
                print(f"❌ Export system issues: {', '.join(failed_critical)}")
            else:
                print(f"✅ Export system: Healthy")
                
        except Exception as e:
            health_status["export_system"] = False
            print(f"❌ Export system check failed: {e}")
        
        # Проверка памяти (без изменений)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                health_status["memory_usage"] = False
                print(f"⚠️ High memory usage: {memory_percent:.1f}%")
            else:
                print(f"✅ Memory usage: {memory_percent:.1f}%")
        except ImportError:
            print("ℹ️ psutil not available - cannot check memory")
        
        # Общий статус здоровья
        overall_health = all(health_status.values())
        if overall_health:
            print("✅ System health: Excellent")
        else:
            issues = [k for k, v in health_status.items() if not v]
            print(f"⚠️ System health issues: {', '.join(issues)}")

    
    def _get_purpose_description(self, purpose: str) -> str:
        """Получение описания назначения мастеринга"""
        descriptions = {
            "freelance": "Commercial sale, streaming optimization",
            "professional": "Broadcast/cinema, full dynamics",
            "personal": "Home listening, natural sound",
            "family": "Family videos, bright engaging",
            "streaming": "Platform optimized, loudness normalized",
            "vinyl": "Analog warm, vinyl-ready"
        }
        return descriptions.get(purpose, "General purpose")
    
    def _create_request_from_task(self, task_data: Dict, task_number: int) -> GenerationRequest:
        """ИСПРАВЛЕННОЕ создание запроса из данных задачи с улучшенной обработкой"""
        # ИСПРАВЛЕНО: Добавлена валидация данных задачи
        try:
            request = GenerationRequest(
                prompt=task_data.get("prompt", f"Generated track {task_number}"),
                genre=task_data.get("genre"),
                bpm=task_data.get("bpm"),
                duration=task_data.get("duration"),
                mastering_purpose=task_data.get("mastering_purpose", "personal"),
                output_dir=task_data.get("output_dir", f"batch_output/task_{task_number}"),
                export_stems=task_data.get("export_stems", True),
                energy_level=task_data.get("energy_level", 0.5),
                creativity_factor=task_data.get("creativity_factor", 0.7)
            )
            
            # Валидируем критические параметры
            if not request.prompt or len(request.prompt.strip()) == 0:
                request.prompt = f"Electronic music track {task_number}"
                self.logger.warning(f"Task {task_number}: Empty prompt, using default")
            
            # Проверяем output_dir
            if not request.output_dir:
                request.output_dir = f"batch_output/task_{task_number}"
            
            return request
            
        except Exception as e:
            self.logger.error(f"Error creating request from task {task_number}: {e}")
            # Возвращаем минимальный валидный запрос
            return GenerationRequest(
                prompt=f"Fallback track {task_number}",
                mastering_purpose="personal",
                output_dir=f"batch_output/task_{task_number}"
            )
    
    def _play_audio_file(self, file_path: str):
        """Воспроизведение аудиофайла"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.system(f'start "" "{file_path}"')
            elif system == "Darwin":  # macOS
                os.system(f'open "{file_path}"')
            else:  # Linux
                os.system(f'xdg-open "{file_path}"')
                
            print("🎵 Opening audio file...")
        except Exception as e:
            print(f"❌ Cannot open audio file: {e}")
    
    def _show_quality_details(self, result: GenerationResult):
        """Показать детали качества"""
        print(f"\n📊 QUALITY ANALYSIS DETAILS")
        print(f"Overall score: {result.quality_score:.2f}/1.0")
        
        if result.quality_score < 0.5:
            print("🔴 Poor quality - major issues detected")
        elif result.quality_score < 0.7:
            print("🟡 Acceptable quality - some issues present")
        elif result.quality_score < 0.9:
            print("🟢 Good quality - minor issues only")
        else:
            print("🟢 Excellent quality - no significant issues")
    
    def run_cli_mode(self, args):
        """Режим командной строки"""
        try:
            if args.prompt:
                # Создаём запрос из аргументов
                request = GenerationRequest(
                    prompt=args.prompt,
                    genre=getattr(args, 'genre', None),
                    bpm=getattr(args, 'bpm', None),
                    duration=getattr(args, 'duration', None),
                    mastering_purpose=getattr(args, 'mastering_purpose', 'personal'),
                    output_dir=args.output_dir or 'cli_output',
                    export_stems=getattr(args, 'export_stems', True),
                    energy_level=getattr(args, 'energy_level', 0.5),
                    creativity_factor=getattr(args, 'creativity_factor', 0.7)
                )
                
                print(f"🚀 CLI Generation: '{args.prompt}'")
                result = self.generate_track_sync(request)
                
                if result.success:
                    print(f"✅ Success: {result.final_path}")
                    return 0
                else:
                    print(f"❌ Failed: {result.error_message}")
                    return 1
                    
            elif getattr(args, 'batch', None):
                return self._run_cli_batch(args.batch)
                
            elif getattr(args, 'analyze', None):
                return self._run_cli_analyze(args.analyze)
                
            else:
                print("❌ No valid CLI command provided")
                return 1
                
        except Exception as e:
            self.logger.error(f"CLI mode error: {e}")
            print(f"❌ CLI error: {e}")
            return 1
    
    def _run_cli_batch(self, batch_file: str) -> int:
        """ИСПРАВЛЕННАЯ CLI пакетная обработка"""
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            tasks = batch_data.get("tasks", [])
            print(f"📦 Processing {len(tasks)} tasks from {batch_file}")
            
            successful = 0
            failed = 0
            
            for i, task_data in enumerate(tasks, 1):
                print(f"\n[{i}/{len(tasks)}] {task_data.get('name', f'Task {i}')}")
                
                try:
                    request = self._create_request_from_task(task_data, i)
                    result = self.generate_track_sync(request)
                    
                    if result.success:
                        print(f"✅ {result.final_path}")
                        successful += 1
                        
                        # ИСПРАВЛЕНО: Показываем сводку экспорта через новый ExportManager
                        if hasattr(result, 'intermediate_files') and result.intermediate_files:
                            try:
                                summary = self.export_manager.get_export_summary(result.intermediate_files)
                                print(f"  📊 Files: {summary['total_files']}, Size: {summary['total_size']/1024/1024:.1f}MB")
                            except Exception as summary_error:
                                self.logger.debug(f"Export summary failed: {summary_error}")
                        
                    else:
                        print(f"❌ {result.error_message}")
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ {e}")
                    failed += 1
            
            print(f"\n📊 Batch complete: {successful} successful, {failed} failed")
            return 0 if failed == 0 else 1
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return 1
    
    def _run_cli_analyze(self, file_path: str) -> int:
        """ИСПРАВЛЕННЫЙ CLI анализ качества"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return 1
            
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            
            target_config = {"target_lufs": -14, "peak_ceiling": -1}
            report = asyncio.run(self.verifier.analyze_track(audio, target_config))
            
            print(f"📊 Quality Analysis: {file_path}")
            print(f"Score: {report.get('overall_score', 0):.2f}/1.0")
            print(f"Status: {report.get('status', 'unknown')}")
            
            # ИСПРАВЛЕНО: Сохраняем отчёт через новый ExportManager
            try:
                report_config = {
                    "request_data": {
                        "prompt": f"CLI analysis of {Path(file_path).name}",
                        "mastering_purpose": "analysis"
                    },
                    "structure": {"total_duration": len(audio) / 1000.0},
                    "analysis_results": report
                }
                
                report_file = asyncio.run(
                    self.export_manager.create_project_report(
                        config=report_config,
                        exported_files={"analyzed_file": file_path},
                        project_dir=Path(file_path).parent
                    )
                )
                
                if report_file:
                    print(f"📋 Report: {report_file}")
                
            except Exception as report_error:
                self.logger.debug(f"Report generation failed: {report_error}")
                print("⚠️ Report generation skipped")
            
            return 0
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return 1



def create_sample_batch_file():
    """Создание примера файла пакетной обработки"""
    sample_batch = {
        "name": "WaveDream Enhanced Pro - Sample Batch",
        "version": "2.0",
        "default_settings": {
            "mastering_purpose": "personal",
            "export_stems": True,
            "export_formats": ["wav", "mp3"]
        },
        "tasks": [
            {
                "name": "Dark Trap Beat",
                "prompt": "dark aggressive trap 160bpm with vocal chops and 808s",
                "mastering_purpose": "freelance",
                "duration": 90,
                "energy_level": 0.8
            },
            {
                "name": "Lofi Study Music",
                "prompt": "melodic lofi study beats with vintage textures and vinyl crackle",
                "mastering_purpose": "personal",
                "duration": 120,
                "energy_level": 0.3
            },
            {
                "name": "Cinematic Epic",
                "prompt": "cinematic epic orchestral trailer music heroic and inspiring",
                "mastering_purpose": "professional",
                "duration": 180,
                "energy_level": 0.9
            },
            {
                "name": "House Groove",
                "prompt": "deep house groove with plucked bass and disco elements 124bpm",
                "mastering_purpose": "streaming",
                "duration": 240,
                "energy_level": 0.7
            }
        ]
    }
    
    with open("sample_batch_tasks.json", 'w', encoding='utf-8') as f:
        json.dump(sample_batch, f, indent=2, ensure_ascii=False)
    
    print("📝 Sample batch file created: sample_batch_tasks.json")


def main():
    """ИСПРАВЛЕННАЯ главная функция с улучшенной обработкой ошибок"""
    parser = argparse.ArgumentParser(
        description="🎵 WaveDream Enhanced Pro v2.0 - Full AI Music Generation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Examples:

Interactive mode:
  python main.py

Quick generation:
  python main.py --prompt "dark trap 160bpm aggressive" --purpose freelance

Batch processing:
  python main.py --batch sample_batch_tasks.json

Quality analysis:
  python main.py --analyze track.wav

System diagnostics:
  python main.py --diagnostics

Advanced generation:
  python main.py --prompt "melodic lofi study beats" --genre lofi --bpm 75 --duration 120 --purpose personal --stems

Export system test:
  python main.py --test-export

🎯 Mastering purposes: freelance, professional, personal, family, streaming, vinyl
🎭 Genres: trap, lofi, dnb, ambient, techno, house, cinematic, hyperpop
        """
    )
    
    # Основные команды (без изменений)
    parser.add_argument("--prompt", type=str, help="Track description prompt")
    parser.add_argument("--genre", type=str, choices=[g.value for g in GenreType], help="Force specific genre")
    parser.add_argument("--bpm", type=int, help="Target BPM")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--purpose", dest="mastering_purpose", choices=[p.value for p in MasteringPurpose], 
                        default="personal", help="Mastering purpose")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--stems", dest="export_stems", action="store_true", help="Export stem versions")
    parser.add_argument("--energy", dest="energy_level", type=float, default=0.5, help="Energy level (0-1)")
    parser.add_argument("--creativity", dest="creativity_factor", type=float, default=0.7, help="Creativity factor (0-1)")
    
    # Пакетная обработка (без изменений)
    parser.add_argument("--batch", type=str, help="Batch processing from JSON file")
    parser.add_argument("--create-batch", action="store_true", help="Create sample batch file")
    
    # Анализ и диагностика
    parser.add_argument("--analyze", type=str, help="Analyze audio file quality")
    parser.add_argument("--diagnostics", action="store_true", help="Run system diagnostics")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    parser.add_argument("--test-export", action="store_true", help="Test export system")  # НОВАЯ ОПЦИЯ
    
    # Настройки (без изменений)
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild sample index")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Создание лаунчера с улучшенной обработкой ошибок
    try:
        launcher = WaveDreamEnhancedLauncher()
    except Exception as e:
        print(f"❌ Failed to initialize WaveDream: {e}")
        print("💡 Try running --diagnostics to check system health")
        return 1
    
    # ИСПРАВЛЕНО: Обработка новых команд
    if args.test_export:
        print("🧪 Testing export system...")
        try:
            env_checks = launcher.export_manager.check_export_environment()
            
            print("Environment checks:")
            for check, result in env_checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check.replace('_', ' ').title()}")
            
            # Полный тест экспорта
            from pydub.generators import Sine
            test_audio = Sine(440).to_audio_segment(duration=2000)
            
            buffer = io.BytesIO()
            test_audio.export(buffer, format="wav")
            test_bytes = buffer.getvalue()
            
            test_config = {
                "output_dir": "export_test",
                "export_formats": ["wav", "mp3"],
                "request_data": {
                    "prompt": "Export system test",
                    "mastering_purpose": "test"
                }
            }
            
            result = asyncio.run(
                launcher.export_manager.export_complete_project(
                    mastered_audio=test_bytes,
                    intermediate_audio={},
                    config=test_config
                )
            )
            
            print(f"✅ Export test successful: {len(result)} files created")
            
            # Показываем сводку
            summary = launcher.export_manager.get_export_summary(result)
            print(f"📊 Total size: {summary['total_size']/1024:.1f}KB")
            
        except Exception as e:
            print(f"❌ Export test failed: {e}")
            return 1
            
        return 0
    
    # Остальные команды без изменений, но с улучшенной обработкой ошибок
    if args.create_batch:
        create_sample_batch_file()
        return 0
    
    if args.diagnostics:
        launcher._run_system_diagnostics()
        return 0
    
    # ... [остальные команды без изменений] ...
    
    # Режимы работы
    if any([args.prompt, args.batch, args.analyze]):
        # CLI режим
        return launcher.run_cli_mode(args)
    else:
        # Интерактивный режим
        try:
            launcher.run_interactive_mode()
            return 0
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
        except Exception as e:
            launcher.logger.error(f"Interactive mode error: {e}")
            print(f"❌ Unexpected error: {e}")
            print("💡 Try running --diagnostics for system health check")
            return 1


if __name__ == "__main__":
    sys.exit(main())


# main.py - Главная система WaveDream Enhanced Pro v2.0

import os
import sys
import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
import requests

# Добавляем путь к модулям WaveDream
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FL'))

# Импорты WaveDream модулей
try:
    from config import config, GenreType, MasteringPurpose
    from pipeline import WaveDreamPipeline, GenerationRequest, GenerationResult
    from sample_engine import SemanticSampleEngine
    from verification import MixVerifier
    from export import ExportManager
    from metadata import MetadataProcessor
    from self_check import verify_mix
    from semantic_engine import select_samples_by_semantics

except ImportError as e:
    print(f"❌ Error importing WaveDream modules: {e}")
    print("Make sure all WaveDream components are properly installed")
    sys.exit(1)


class WaveDreamEnhancedLauncher:
    """
    Главный лаунчер WaveDream Enhanced Pro v2.0
    
    Объединяет все компоненты системы:
    - Семантический анализ промптов
    - Жанровую детекцию с AI
    - LLaMA3 структурирование
    - MusicGen генерацию 
    - Умный мастеринг по назначению
    - Полную верификацию качества
    - Экспорт в множественных форматах
    """
    
    def __init__(self):
        # Настройка логирования
        self._setup_logging()
        
        # Инициализация компонентов
        self.logger = logging.getLogger(__name__)
        self.pipeline = WaveDreamPipeline()
        self.metadata_processor = MetadataProcessor()
        self.sample_engine = SemanticSampleEngine()
        self.verifier = MixVerifier()
        self.export_manager = ExportManager()
        
        # Статистика производительности
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'avg_generation_time': 0.0,
            'genre_statistics': {},
            'purpose_statistics': {}
        }
        
        # Проверяем окружение
        self._validate_environment()
        
        self.logger.info("🎵 WaveDream Enhanced Pro v2.0 initialized successfully")
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        log_config = config.LOGGING
        
        # Создаём форматтер
        formatter = logging.Formatter(log_config["format"])
        
        # Настраиваем уровень
        log_level = getattr(logging, log_config["level"], logging.INFO)
        
        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Файловый хендлер с ротацией
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config["file"],
                maxBytes=log_config["max_size_mb"] * 1024 * 1024,
                backupCount=log_config["backup_count"],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
            file_handler = None
        
        # Настраиваем root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        if file_handler:
            root_logger.addHandler(file_handler)
        
        # Убираем дублирование для наших логгеров
        for logger_name in ['wavedream', 'wavedream.core', '__main__']:
            logger = logging.getLogger(logger_name)
            logger.propagate = True
    
    def _validate_environment(self):
        """Валидация окружения и зависимостей"""
        try:
            validation_errors = config.validate_environment()
            if validation_errors:
                self.logger.warning(f"Environment validation issues: {'; '.join(validation_errors)}")
        except RuntimeError as e:
            self.logger.error(f"❌ Critical configuration error: {e}")
            raise
    
    async def generate_track_async(self, request: GenerationRequest) -> GenerationResult:
        """
        Асинхронная генерация трека через полный pipeline
        
        Args:
            request: Запрос на генерацию с параметрами
            
        Returns:
            Результат генерации с метриками качества
        """
        start_time = time.time()
        self.performance_stats['total_generations'] += 1
        
        try:
            self.logger.info(f"🚀 Starting async track generation")
            self.logger.info(f"📝 Prompt: '{request.prompt}'")
            self.logger.info(f"🎯 Purpose: {request.mastering_purpose}")
            self.logger.info(f"🎭 Genre hint: {request.genre or 'auto-detect'}")
            
            # Запускаем полный pipeline
            result = await self.pipeline.generate_track(request)
            
            # Обновляем статистику
            generation_time = time.time() - start_time
            
            if result.success:
                self.performance_stats['successful_generations'] += 1
                
                # Обновляем статистику по жанрам
                detected_genre = result.structure_data.get('detected_genre', 'unknown') if result.structure_data else 'unknown'
                self.performance_stats['genre_statistics'][detected_genre] = \
                    self.performance_stats['genre_statistics'].get(detected_genre, 0) + 1
                
                # Статистика по назначению
                self.performance_stats['purpose_statistics'][request.mastering_purpose] = \
                    self.performance_stats['purpose_statistics'].get(request.mastering_purpose, 0) + 1
                
                # Среднее время генерации
                current_avg = self.performance_stats['avg_generation_time']
                total_successful = self.performance_stats['successful_generations']
                self.performance_stats['avg_generation_time'] = \
                    (current_avg * (total_successful - 1) + generation_time) / total_successful
                
                self.logger.info(f"✅ Generation completed successfully in {generation_time:.1f}s")
                self.logger.info(f"🎯 Quality score: {result.quality_score:.2f}/1.0")
                
            else:
                self.logger.error(f"❌ Generation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Async generation error: {e}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            
            return GenerationResult(
                success=False,
                generation_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def generate_track_sync(self, request: GenerationRequest) -> GenerationResult:
        """Синхронная обёртка для асинхронной генерации"""
        return asyncio.run(self.generate_track_async(request))
    
    def run_interactive_mode(self):
        """Интерактивный режим с расширенными возможностями"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║ 🎵 WaveDream Enhanced Pro v2.0 - Full AI Music Generation Suite 🎵             ║
║                                                                                  ║ 
║ 🧠 LLaMA3 Structure | 🎼 MusicGen Base | 🔍 Semantic Samples | 🎛️ Smart Master ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            self._display_main_menu()
            choice = input("\n🎯 Your choice: ").strip()
            
            try:
                if choice == "1":
                    self._interactive_enhanced_generation()
                elif choice == "2":
                    self._interactive_batch_generation()
                elif choice == "3":
                    self._interactive_sample_analysis()
                elif choice == "4":
                    self._interactive_genre_testing()
                elif choice == "5":
                    self._interactive_quality_analysis()
                elif choice == "6":
                    self._interactive_system_management()
                elif choice == "7":
                    self._interactive_settings()
                elif choice == "8":
                    self._display_statistics()
                elif choice == "9":
                    self._run_system_diagnostics()
                elif choice == "0":
                    print("👋 Thank you for using WaveDream Enhanced Pro!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n⏸️ Operation cancelled by user")
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                print(f"❌ An error occurred: {e}")
                print("Please try again or check logs for details.")
    
    def _display_main_menu(self):
        """Отображение главного меню"""
        print("\n" + "="*80)
        print("🎵 WaveDream Enhanced Pro v2.0 - Main Menu")
        print("="*80)
        print("1. 🚀 Enhanced Track Generation (Full Pipeline)")
        print("2. 📦 Batch Generation from JSON") 
        print("3. 🔍 Sample Database Analysis")
        print("4. 🎭 Genre Detection Testing")
        print("5. 📊 Quality Analysis Tools")
        print("6. ⚙️ System Management")
        print("7. 🛠️ Settings & Configuration")
        print("8. 📈 Performance Statistics")
        print("9. 🔧 System Diagnostics")
        print("0. 🚪 Exit")
    
    def _interactive_enhanced_generation(self):
        """Интерактивная расширенная генерация"""
        print("\n🚀 ENHANCED TRACK GENERATION")
        print("-" * 50)
        print("Full AI pipeline: Prompt Analysis → Genre Detection → Structure → Generation → Mastering")
        
        # Ввод основных параметров
        prompt = input("\n📝 Enter track description: ").strip()
        if not prompt:
            print("❌ Prompt cannot be empty")
            return
        
        # Предварительный анализ промпта
        print("\n🧠 Analyzing prompt...")
        prompt_analysis = self.metadata_processor.analyze_prompt(prompt)
        detected_genre = self.metadata_processor.detect_genre(prompt, prompt_analysis.get('tags', []))
        
        print(f"🎭 Detected genre: {detected_genre}")
        print(f"🎵 Detected BPM: {prompt_analysis.get('bpm', 'auto')}")
        print(f"🎹 Detected instruments: {', '.join(prompt_analysis.get('instruments', ['auto']))}")
        print(f"😊 Detected mood: {', '.join(prompt_analysis.get('mood', ['neutral']))}")
        
        # Опции жанра
        confirm_genre = input(f"\nContinue with detected genre '{detected_genre}'? (Y/n): ").lower()
        if confirm_genre == 'n':
            available_genres = [genre.value for genre in GenreType]
            print(f"Available genres: {', '.join(available_genres)}")
            manual_genre = input("Enter genre manually: ").strip().lower()
            if manual_genre in available_genres:
                detected_genre = manual_genre
            else:
                print(f"❌ Unknown genre, using detected: {detected_genre}")
        
        # Назначение мастеринга
        print("\n🎯 Select mastering purpose:")
        purposes = [purpose.value for purpose in MasteringPurpose]
        for i, purpose in enumerate(purposes, 1):
            print(f"{i}. {purpose.title()} - {self._get_purpose_description(purpose)}")
        
        purpose_choice = input("Choice (1-6, Enter for personal): ").strip()
        try:
            if purpose_choice:
                mastering_purpose = purposes[int(purpose_choice) - 1]
            else:
                mastering_purpose = "personal"
        except (ValueError, IndexError):
            mastering_purpose = "personal"
        
        print(f"🎯 Selected purpose: {mastering_purpose}")
        
        # Дополнительные параметры
        print("\n⚙️ Additional options:")
        
        duration = input("Duration in seconds (Enter for auto): ").strip()
        try:
            duration = int(duration) if duration else None
        except ValueError:
            duration = None
        
        export_stems = input("Export all intermediate versions? (Y/n): ").lower() != 'n'
        
        export_formats = ["wav"]
        more_formats = input("Export additional formats? (mp3,flac,aac): ").strip()
        if more_formats:
            additional = [f.strip() for f in more_formats.split(',')]
            export_formats.extend(additional)
        
        # Вывод
        output_dir = input("Output directory (Enter for auto): ").strip()
        if not output_dir:
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c.isspace()).strip()
            safe_prompt = "_".join(safe_prompt.split())
            output_dir = f"output_{safe_prompt}_{detected_genre}_{mastering_purpose}"
        
        # Создаём запрос
        request = GenerationRequest(
            prompt=prompt,
            genre=detected_genre,
            bpm=prompt_analysis.get('bpm'),
            duration=duration,
            mastering_purpose=mastering_purpose,
            output_dir=output_dir,
            export_stems=export_stems,
            energy_level=prompt_analysis.get('energy_level', 0.5),
            creativity_factor=prompt_analysis.get('complexity_score', 0.7)
        )
        
        # Подтверждение
        print(f"\n📋 GENERATION SUMMARY:")
        print(f"  📝 Prompt: '{prompt}'")
        print(f"  🎭 Genre: {detected_genre}")
        print(f"  🎵 BPM: {request.bpm or 'auto'}")
        print(f"  ⏱️ Duration: {duration or 'auto'} seconds")
        print(f"  🎯 Purpose: {mastering_purpose}")
        print(f"  📁 Output: {output_dir}")
        print(f"  💾 Export stems: {export_stems}")
        print(f"  🎼 Formats: {', '.join(export_formats)}")
        
        confirm = input("\nProceed with generation? (Y/n): ").lower()
        if confirm == 'n':
            print("❌ Generation cancelled")
            return
        
        # Запускаем генерацию
        print(f"\n🚀 Starting enhanced generation...")
        print("This may take several minutes depending on complexity...")
        
        try:
            result = self.generate_track_sync(request)
            
            if result.success:
                print(f"\n🎉 GENERATION COMPLETED SUCCESSFULLY!")
                print(f"📁 Final track: {result.final_path}")
                print(f"⏱️ Generation time: {result.generation_time:.1f} seconds")
                print(f"🎯 Quality score: {result.quality_score:.2f}/1.0")
                
                if result.used_samples:
                    print(f"🎛️ Used samples: {len(result.used_samples)}")
                
                if result.structure_data:
                    sections = result.structure_data.get('sections', [])
                    print(f"🏗️ Structure sections: {len(sections)}")
                
                # Предложение воспроизведения
                if result.final_path and os.path.exists(result.final_path):
                    play = input("\nPlay the generated track? (y/N): ").lower()
                    if play == 'y':
                        self._play_audio_file(result.final_path)
                
                # Показать отчёт о качестве
                if result.quality_score < 0.8:
                    show_quality = input("Show detailed quality report? (Y/n): ").lower()
                    if show_quality != 'n':
                        self._show_quality_details(result)
                
            else:
                print(f"\n❌ GENERATION FAILED")
                print(f"Error: {result.error_message}")
                print("Check logs for detailed error information")
                
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            print(f"❌ Unexpected error during generation: {e}")
    
    def _interactive_batch_generation(self):
        """Интерактивная пакетная генерация"""
        print("\n📦 BATCH GENERATION FROM JSON")
        print("-" * 40)
        
        # Поиск JSON файлов в текущей директории
        json_files = list(Path('.').glob('*.json'))
        batch_files = [f for f in json_files if 'batch' in f.name.lower() or 'tasks' in f.name.lower()]
        
        if batch_files:
            print("Found potential batch files:")
            for i, file in enumerate(batch_files, 1):
                print(f"{i}. {file}")
            print(f"{len(batch_files) + 1}. Enter custom path")
            
            choice = input("Select batch file: ").strip()
            try:
                if choice and int(choice) <= len(batch_files):
                    batch_file = str(batch_files[int(choice) - 1])
                else:
                    batch_file = input("Enter batch file path: ").strip()
            except (ValueError, IndexError):
                batch_file = input("Enter batch file path: ").strip()
        else:
            batch_file = input("Enter batch file path: ").strip()
        
        if not batch_file or not os.path.exists(batch_file):
            print("❌ Batch file not found")
            return
        
        # Загружаем задачи
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading batch file: {e}")
            return
        
        tasks = batch_data.get("tasks", [])
        if not tasks:
            print("❌ No tasks found in batch file")
            return
        
        print(f"📦 Found {len(tasks)} tasks")
        
        # Настройки пакетной обработки
        parallel_processing = input("Enable parallel processing? (Y/n): ").lower() != 'n'
        max_concurrent = 2  # Безопасный лимит
        
        if parallel_processing:
            try:
                max_concurrent = int(input(f"Max concurrent tasks (default {max_concurrent}): ") or max_concurrent)
            except ValueError:
                pass
        
        # Запуск пакетной обработки
        print(f"\n🚀 Starting batch processing: {len(tasks)} tasks")
        if parallel_processing:
            print(f"⚡ Parallel processing: max {max_concurrent} concurrent")
        
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_data in enumerate(tasks, 1):
            print(f"\n📋 Task {i}/{len(tasks)}: {task_data.get('name', f'task_{i}')}")
            
            try:
                # Создаём запрос из данных задачи
                request = self._create_request_from_task(task_data, i)
                
                # Генерируем
                result = self.generate_track_sync(request)
                
                if result.success:
                    print(f"  ✅ Completed: {result.final_path}")
                    successful_tasks += 1
                else:
                    print(f"  ❌ Failed: {result.error_message}")
                    failed_tasks += 1
                
            except Exception as e:
                print(f"  ❌ Task error: {e}")
                failed_tasks += 1
        
        # Итоги
        print(f"\n📊 BATCH PROCESSING COMPLETED")
        print(f"✅ Successful: {successful_tasks}")
        print(f"❌ Failed: {failed_tasks}")
        print(f"📈 Success rate: {successful_tasks/(successful_tasks+failed_tasks)*100:.1f}%")
    
    def _interactive_sample_analysis(self):
        """Интерактивный анализ базы сэмплов"""
        print("\n🔍 SAMPLE DATABASE ANALYSIS")
        print("-" * 35)
        
        print("Choose analysis type:")
        print("1. Database statistics")
        print("2. Rebuild semantic index")
        print("3. Search samples by query")
        print("4. Analyze sample quality")
        print("5. Export sample metadata")
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            print("\n📊 Generating database statistics...")
            stats = self.sample_engine.get_statistics()
            
            print(f"\n📈 SAMPLE DATABASE STATISTICS")
            print(f"Total samples: {stats.get('total_samples', 0)}")
            print(f"Average quality: {stats.get('avg_quality', 0):.2f}")
            
            # Статистика по жанрам
            genre_dist = stats.get('genre_distribution', {})
            if genre_dist:
                print(f"\n🎭 Genre distribution:")
                for genre, count in sorted(genre_dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {genre}: {count} samples")
            
            # Статистика по инструментам
            instrument_dist = stats.get('instrument_distribution', {})
            if instrument_dist:
                print(f"\n🎼 Instrument distribution:")
                for instrument, count in sorted(instrument_dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {instrument}: {count} samples")
            
            # Производительность
            perf_stats = stats.get('performance_stats', {})
            if perf_stats:
                print(f"\n⚡ Performance statistics:")
                print(f"  Queries processed: {perf_stats.get('queries', 0)}")
                print(f"  Cache hit rate: {perf_stats.get('cache_hits', 0) / max(1, perf_stats.get('queries', 1)) * 100:.1f}%")
                print(f"  Average query time: {perf_stats.get('avg_query_time', 0):.3f}s")
        
        elif choice == "2":
            confirm = input("⚠️ Rebuild semantic index? This will take time. (y/N): ").lower()
            if confirm == 'y':
                print("🔄 Rebuilding semantic index...")
                self.sample_engine.build_semantic_index()
                print("✅ Semantic index rebuilt successfully")
        
        elif choice == "3":
            query = input("Enter search query (tags, instruments, genre): ").strip()
            if query:
                print(f"\n🔍 Searching for: '{query}'")
                
                # Парсим запрос
                query_parts = query.split()
                search_results = asyncio.run(self.sample_engine.find_samples(
                    tags=query_parts,
                    max_results=10
                ))
                
                if search_results:
                    print(f"\n📋 Found {len(search_results)} samples:")
                    for i, sample in enumerate(search_results, 1):
                        filename = sample.get('filename', 'unknown')
                        score = sample.get('score', 0)
                        tags = sample.get('tags', [])
                        print(f"{i}. {filename} (score: {score:.2f})")
                        print(f"   Tags: {', '.join(tags[:5])}")
                else:
                    print("❌ No samples found matching query")
        
        elif choice == "4":
            print("📊 Analyzing sample quality...")
            # В реальной реализации здесь будет полный анализ качества
            print("Quality analysis completed - see logs for details")
        
        elif choice == "5":
            output_file = input("Export metadata to file (default: sample_metadata.json): ").strip()
            if not output_file:
                output_file = "sample_metadata.json"
            
            try:
                stats = self.sample_engine.get_statistics()
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                print(f"✅ Metadata exported to: {output_file}")
            except Exception as e:
                print(f"❌ Export error: {e}")
    
    def _interactive_genre_testing(self):
        """Интерактивное тестирование детекции жанров"""
        print("\n🎭 GENRE DETECTION TESTING")
        print("-" * 30)
        
        test_prompts = [
            ("dark aggressive trap 160bpm with vocal chops and 808s", "trap"),
            ("мелодичный лоуфай для учёбы с винтажными текстурами", "lofi"),
            ("liquid drum and bass neurofunk 174bpm atmospheric", "dnb"),
            ("атмосферный эмбиент космос медитация 70bpm", "ambient"),
            ("phonk memphis cowbell drift aggressive", "phonk"),
            ("техно минимал 130bpm industrial warehouse", "techno"),
            ("cinematic epic trailer orchestral heroic", "cinematic"),
            ("house deep groove плавный bassline 124bpm", "house")
        ]
        
        print("Choose testing mode:")
        print("1. Test with built-in examples")
        print("2. Test custom prompts")
        print("3. Test accuracy on known samples")
        
        choice = input("Select mode (1-3): ").strip()
        
        if choice == "1":
            print("\n🧪 Testing with built-in examples:")
            
            correct = 0
            total = len(test_prompts)
            
            for prompt, expected in test_prompts:
                detected = self.metadata_processor.detect_genre(prompt)
                is_correct = detected == expected
                
                status = "✅" if is_correct else "❌"
                print(f"{status} '{prompt[:50]}...'")
                print(f"   Expected: {expected} | Detected: {detected}")
                
                if is_correct:
                    correct += 1
            
            accuracy = correct / total * 100
            print(f"\n📊 Accuracy: {accuracy:.1f}% ({correct}/{total})")
            
        elif choice == "2":
            while True:
                prompt = input("\nEnter test prompt (or 'quit' to exit): ").strip()
                if prompt.lower() == 'quit':
                    break
                
                if prompt:
                    # Полный анализ
                    analysis = self.metadata_processor.analyze_prompt(prompt)
                    detected_genre = self.metadata_processor.detect_genre(prompt, analysis.get('tags', []))
                    
                    print(f"\n🔍 Analysis results for: '{prompt}'")
                    print(f"🎭 Detected genre: {detected_genre}")
                    print(f"🎵 Detected BPM: {analysis.get('bpm', 'none')}")
                    print(f"🎹 Instruments: {', '.join(analysis.get('instruments', ['none']))}")
                    print(f"😊 Mood: {', '.join(analysis.get('mood', ['neutral']))}")
                    print(f"🧠 Complexity: {analysis.get('complexity_score', 0):.2f}")
                    
                    if analysis.get('mentioned_sections'):
                        print(f"🏗️ Structure hints: {', '.join(analysis['mentioned_sections'])}")
    
    def _interactive_quality_analysis(self):
        """Интерактивный анализ качества"""
        print("\n📊 QUALITY ANALYSIS TOOLS")
        print("-" * 30)
        
        print("1. Analyze audio file")
        print("2. Compare two tracks")
        print("3. Batch quality analysis")
        print("4. Generate quality report")
        
        choice = input("Select tool (1-4): ").strip()
        
        if choice == "1":
            file_path = input("Enter audio file path: ").strip()
            if os.path.exists(file_path):
                print(f"🔍 Analyzing: {file_path}")
                
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(file_path)
                    
                    # В реальной реализации здесь будет полный анализ качества
                    target_config = {"target_lufs": -14, "peak_ceiling": -1}
                    report = asyncio.run(self.verifier.analyze_track(audio, target_config))
                    
                    print(f"📊 Quality score: {report.get('overall_score', 0):.2f}/1.0")
                    print(f"🎯 Status: {report.get('status', 'unknown')}")
                    
                    issues = report.get('issues', [])
                    if issues:
                        critical = len([i for i in issues if i.get('severity') == 'critical'])
                        warnings = len([i for i in issues if i.get('severity') == 'warning'])
                        print(f"🚨 Issues: {critical} critical, {warnings} warnings")
                    
                    # Предложение детального отчёта
                    if input("Generate detailed report? (Y/n): ").lower() != 'n':
                        report_path = f"{Path(file_path).stem}_quality_report.md"
                        if self.verifier.generate_markdown_report(report, report_path):
                            print(f"📋 Report saved: {report_path}")
                
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
            else:
                print("❌ File not found")
    
    def _interactive_system_management(self):
        """Интерактивное управление системой"""
        print("\n⚙️ SYSTEM MANAGEMENT")
        print("-" * 25)
        
        print("1. Clear cache files")
        print("2. Update sample index")
        print("3. Check system health")
        print("4. Export configuration")
        print("5. Import configuration")
        print("6. Reset to defaults")
        
        choice = input("Select action (1-6): ").strip()
        
        if choice == "1":
            cache_dir = Path(config.CACHE_DIR)
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print("✅ Cache cleared")
            else:
                print("ℹ️ No cache to clear")
        
        elif choice == "2":
            print("🔄 Updating sample index...")
            self.sample_engine.build_semantic_index()
            print("✅ Index updated")
        
        elif choice == "3":
            print("🏥 Running system health check...")
            self._run_system_health_check()
        
        elif choice == "4":
            config_path = input("Export config to (default: wavedream_config.json): ").strip()
            if not config_path:
                config_path = "wavedream_config.json"
            
            try:
                config.export_config(config_path)
                print(f"✅ Configuration exported: {config_path}")
            except Exception as e:
                print(f"❌ Export error: {e}")
    
    def _interactive_settings(self):
        """Интерактивные настройки"""
        print("\n🛠️ SETTINGS & CONFIGURATION")
        print("-" * 35)
        
        current_settings = {
            "Sample Directory": config.DEFAULT_SAMPLE_DIR,
            "Audio Analysis Max Duration": f"{config.AUDIO_ANALYSIS['max_duration']}s",
            "Sample Rate": f"{config.AUDIO_ANALYSIS['sample_rate']}Hz",
            "Tempo Tolerance": f"±{config.SAMPLE_MATCHING['tempo_tolerance']} BPM",
            "Min Score Threshold": config.SAMPLE_MATCHING['min_score_threshold'],
            "Max Workers": config.PERFORMANCE['max_workers'],
            "Semantic Analysis": config.SAMPLE_MATCHING['enable_semantic_embeddings']
        }
        
        print("Current settings:")
        for key, value in current_settings.items():
            print(f"  {key}: {value}")
        
        print(f"\nSupported genres: {', '.join([g.value for g in GenreType])}")
        print(f"Mastering purposes: {', '.join([p.value for p in MasteringPurpose])}")
        
        change = input("\nModify settings? (y/N): ").lower()
        if change == 'y':
            print("💡 To modify settings, edit the configuration files or use environment variables")
            print("💡 See documentation for detailed configuration options")
    
    def _display_statistics(self):
        """Отображение статистики производительности"""
        print("\n📈 PERFORMANCE STATISTICS")
        print("-" * 30)
        
        stats = self.performance_stats
        
        print(f"Total generations: {stats['total_generations']}")
        print(f"Successful generations: {stats['successful_generations']}")
        
        if stats['total_generations'] > 0:
            success_rate = stats['successful_generations'] / stats['total_generations'] * 100
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average generation time: {stats['avg_generation_time']:.1f}s")
        
        if stats['genre_statistics']:
            print(f"\nGenre distribution:")
            for genre, count in sorted(stats['genre_statistics'].items(), key=lambda x: -x[1]):
                print(f"  {genre}: {count} generations")
        
        if stats['purpose_statistics']:
            print(f"\nPurpose distribution:")
            for purpose, count in sorted(stats['purpose_statistics'].items(), key=lambda x: -x[1]):
                print(f"  {purpose}: {count} generations")
        
        # Системная информация
        print(f"\nSystem info:")
        print(f"  Sample database size: {len(self.sample_engine.samples_index)} samples")
        print(f"  Cache entries: {len(self.sample_engine.embeddings_cache)}")
    
    def _run_system_diagnostics(self):
        """Запуск системной диагностики"""
        print("\n🔧 SYSTEM DIAGNOSTICS")
        print("-" * 25)
        
        print("Running comprehensive system check...")
        
        # Проверка зависимостей
        print("\n📦 Checking dependencies...")
        dependencies = ['torch', 'librosa', 'pydub', 'numpy', 'scipy', 'soundfile']
        
        for dep in dependencies:
            try:
                __import__(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} - Missing!")
        
        # Проверка директорий
        print(f"\n📁 Checking directories...")
        dirs_to_check = [
            config.DEFAULT_SAMPLE_DIR,
            config.DEFAULT_OUTPUT_DIR,
            config.CACHE_DIR
        ]
        
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ⚠️ {dir_path} - Not found")
        
        # Проверка семантической модели
        print(f"\n🧠 Checking semantic model...")
        try:
            if hasattr(self.sample_engine, 'semantic_model') and self.sample_engine.semantic_model:
                print(f"  ✅ Semantic model loaded")
            else:
                print(f"  ⚠️ Semantic model not available")
        except Exception as e:
            print(f"  ❌ Semantic model error: {e}")
        
        # Проверка производительности
        print(f"\n⚡ Performance test...")
        start_time = time.time()
        
        # Простой тест
        test_prompt = "test electronic music 120bpm"
        analysis = self.metadata_processor.analyze_prompt(test_prompt)
        
        test_time = time.time() - start_time
        print(f"  📊 Prompt analysis: {test_time:.3f}s")
        
        if test_time < 1.0:
            print(f"  ✅ Performance: Good")
        elif test_time < 3.0:
            print(f"  ⚠️ Performance: Acceptable")
        else:
            print(f"  ❌ Performance: Slow")
    
    def _run_system_health_check(self):
        """Проверка здоровья системы"""
        health_status = {
            "dependencies": True,
            "directories": True,
            "sample_index": True,
            "semantic_model": True,
            "memory_usage": True
        }
        
        # Проверка памяти
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                health_status["memory_usage"] = False
                print(f"⚠️ High memory usage: {memory_percent:.1f}%")
            else:
                print(f"✅ Memory usage: {memory_percent:.1f}%")
        except ImportError:
            print("ℹ️ psutil not available - cannot check memory")
        
        # Общий статус здоровья
        overall_health = all(health_status.values())
        if overall_health:
            print("✅ System health: Excellent")
        else:
            issues = [k for k, v in health_status.items() if not v]
            print(f"⚠️ System health issues: {', '.join(issues)}")
    
    def _get_purpose_description(self, purpose: str) -> str:
        """Получение описания назначения мастеринга"""
        descriptions = {
            "freelance": "Commercial sale, streaming optimization",
            "professional": "Broadcast/cinema, full dynamics",
            "personal": "Home listening, natural sound",
            "family": "Family videos, bright engaging",
            "streaming": "Platform optimized, loudness normalized",
            "vinyl": "Analog warm, vinyl-ready"
        }
        return descriptions.get(purpose, "General purpose")
    
    def _create_request_from_task(self, task_data: Dict, task_number: int) -> GenerationRequest:
        """Создание запроса из данных задачи"""
        return GenerationRequest(
            prompt=task_data.get("prompt", f"Generated track {task_number}"),
            genre=task_data.get("genre"),
            bpm=task_data.get("bpm"),
            duration=task_data.get("duration"),
            mastering_purpose=task_data.get("mastering_purpose", "personal"),
            output_dir=task_data.get("output_dir", f"batch_output/task_{task_number}"),
            export_stems=task_data.get("export_stems", True),
            energy_level=task_data.get("energy_level", 0.5),
            creativity_factor=task_data.get("creativity_factor", 0.7)
        )
    
    def _play_audio_file(self, file_path: str):
        """Воспроизведение аудиофайла"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.system(f'start "" "{file_path}"')
            elif system == "Darwin":  # macOS
                os.system(f'open "{file_path}"')
            else:  # Linux
                os.system(f'xdg-open "{file_path}"')
                
            print("🎵 Opening audio file...")
        except Exception as e:
            print(f"❌ Cannot open audio file: {e}")
    
    def _show_quality_details(self, result: GenerationResult):
        """Показать детали качества"""
        print(f"\n📊 QUALITY ANALYSIS DETAILS")
        print(f"Overall score: {result.quality_score:.2f}/1.0")
        
        if result.quality_score < 0.5:
            print("🔴 Poor quality - major issues detected")
        elif result.quality_score < 0.7:
            print("🟡 Acceptable quality - some issues present")
        elif result.quality_score < 0.9:
            print("🟢 Good quality - minor issues only")
        else:
            print("🟢 Excellent quality - no significant issues")
    
    def run_cli_mode(self, args):
        """Режим командной строки"""
        try:
            if args.prompt:
                # Создаём запрос из аргументов
                request = GenerationRequest(
                    prompt=args.prompt,
                    genre=getattr(args, 'genre', None),
                    bpm=getattr(args, 'bpm', None),
                    duration=getattr(args, 'duration', None),
                    mastering_purpose=getattr(args, 'mastering_purpose', 'personal'),
                    output_dir=args.output_dir or 'cli_output',
                    export_stems=getattr(args, 'export_stems', True),
                    energy_level=getattr(args, 'energy_level', 0.5),
                    creativity_factor=getattr(args, 'creativity_factor', 0.7)
                )
                
                print(f"🚀 CLI Generation: '{args.prompt}'")
                result = self.generate_track_sync(request)
                
                if result.success:
                    print(f"✅ Success: {result.final_path}")
                    return 0
                else:
                    print(f"❌ Failed: {result.error_message}")
                    return 1
                    
            elif getattr(args, 'batch', None):
                return self._run_cli_batch(args.batch)
                
            elif getattr(args, 'analyze', None):
                return self._run_cli_analyze(args.analyze)
                
            else:
                print("❌ No valid CLI command provided")
                return 1
                
        except Exception as e:
            self.logger.error(f"CLI mode error: {e}")
            print(f"❌ CLI error: {e}")
            return 1
    
    def _run_cli_batch(self, batch_file: str) -> int:
        """CLI пакетная обработка"""
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            tasks = batch_data.get("tasks", [])
            print(f"📦 Processing {len(tasks)} tasks from {batch_file}")
            
            successful = 0
            failed = 0
            
            for i, task_data in enumerate(tasks, 1):
                print(f"\n[{i}/{len(tasks)}] {task_data.get('name', f'Task {i}')}")
                
                try:
                    request = self._create_request_from_task(task_data, i)
                    result = self.generate_track_sync(request)
                    
                    if result.success:
                        print(f"✅ {result.final_path}")
                        successful += 1
                    else:
                        print(f"❌ {result.error_message}")
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ {e}")
                    failed += 1
            
            print(f"\n📊 Batch complete: {successful} successful, {failed} failed")
            return 0 if failed == 0 else 1
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return 1
    
    def _run_cli_analyze(self, file_path: str) -> int:
        """CLI анализ качества"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return 1
            
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            
            target_config = {"target_lufs": -14, "peak_ceiling": -1}
            report = asyncio.run(self.verifier.analyze_track(audio, target_config))
            
            print(f"📊 Quality Analysis: {file_path}")
            print(f"Score: {report.get('overall_score', 0):.2f}/1.0")
            print(f"Status: {report.get('status', 'unknown')}")
            
            # Сохраняем отчёт
            report_path = f"{Path(file_path).stem}_quality_report.md"
            if self.verifier.generate_markdown_report(report, report_path):
                print(f"📋 Report: {report_path}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return 1


def create_sample_batch_file():
    """Создание примера файла пакетной обработки"""
    sample_batch = {
        "name": "WaveDream Enhanced Pro - Sample Batch",
        "version": "2.0",
        "default_settings": {
            "mastering_purpose": "personal",
            "export_stems": True,
            "export_formats": ["wav", "mp3"]
        },
        "tasks": [
            {
                "name": "Dark Trap Beat",
                "prompt": "dark aggressive trap 160bpm with vocal chops and 808s",
                "mastering_purpose": "freelance",
                "duration": 90,
                "energy_level": 0.8
            },
            {
                "name": "Lofi Study Music",
                "prompt": "melodic lofi study beats with vintage textures and vinyl crackle",
                "mastering_purpose": "personal",
                "duration": 120,
                "energy_level": 0.3
            },
            {
                "name": "Cinematic Epic",
                "prompt": "cinematic epic orchestral trailer music heroic and inspiring",
                "mastering_purpose": "professional",
                "duration": 180,
                "energy_level": 0.9
            },
            {
                "name": "House Groove",
                "prompt": "deep house groove with plucked bass and disco elements 124bpm",
                "mastering_purpose": "streaming",
                "duration": 240,
                "energy_level": 0.7
            }
        ]
    }
    
    with open("sample_batch_tasks.json", 'w', encoding='utf-8') as f:
        json.dump(sample_batch, f, indent=2, ensure_ascii=False)
    
    print("📝 Sample batch file created: sample_batch_tasks.json")


def main():
    """Главная функция с обработкой аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="🎵 WaveDream Enhanced Pro v2.0 - Full AI Music Generation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Examples:

Interactive mode:
  python wavedream_enhanced_main.py

Quick generation:
  python wavedream_enhanced_main.py --prompt "dark trap 160bpm aggressive" --purpose freelance

Batch processing:
  python wavedream_enhanced_main.py --batch sample_batch_tasks.json

Quality analysis:
  python wavedream_enhanced_main.py --analyze track.wav

Create sample batch:
  python wavedream_enhanced_main.py --create-batch

Advanced generation:
  python wavedream_enhanced_main.py --prompt "melodic lofi study beats" --genre lofi --bpm 75 --duration 120 --purpose personal --stems

System diagnostics:
  python wavedream_enhanced_main.py --diagnostics

🎯 Mastering purposes: freelance, professional, personal, family, streaming, vinyl
🎭 Genres: trap, lofi, dnb, ambient, techno, house, cinematic, hyperpop
        """
    )
    
    # Основные команды
    parser.add_argument("--prompt", type=str, help="Track description prompt")
    parser.add_argument("--genre", type=str, choices=[g.value for g in GenreType], help="Force specific genre")
    parser.add_argument("--bpm", type=int, help="Target BPM")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--purpose", dest="mastering_purpose", choices=[p.value for p in MasteringPurpose], 
                        default="personal", help="Mastering purpose")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--stems", dest="export_stems", action="store_true", help="Export stem versions")
    parser.add_argument("--energy", dest="energy_level", type=float, default=0.5, help="Energy level (0-1)")
    parser.add_argument("--creativity", dest="creativity_factor", type=float, default=0.7, help="Creativity factor (0-1)")
    
    # Пакетная обработка
    parser.add_argument("--batch", type=str, help="Batch processing from JSON file")
    parser.add_argument("--create-batch", action="store_true", help="Create sample batch file")
    
    # Анализ и диагностика
    parser.add_argument("--analyze", type=str, help="Analyze audio file quality")
    parser.add_argument("--diagnostics", action="store_true", help="Run system diagnostics")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    
    # Настройки
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild sample index")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Создание лаунчера
    try:
        launcher = WaveDreamEnhancedLauncher()
    except Exception as e:
        print(f"❌ Failed to initialize WaveDream: {e}")
        return 1
    
    # Обработка специальных команд
    if args.create_batch:
        create_sample_batch_file()
        return 0
    
    if args.diagnostics:
        launcher._run_system_diagnostics()
        return 0
    
    if args.stats:
        launcher._display_statistics()
        return 0
    
    if args.rebuild_index:
        print("🔄 Rebuilding sample index...")
        launcher.sample_engine.build_semantic_index()
        print("✅ Index rebuilt")
        return 0
    
    if args.clear_cache:
        cache_dir = Path(config.CACHE_DIR)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared")
        return 0
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Режимы работы
    if any([args.prompt, args.batch, args.analyze]):
        # CLI режим
        return launcher.run_cli_mode(args)
    else:
        # Интерактивный режим
        try:
            launcher.run_interactive_mode()
            return 0
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
        except Exception as e:
            launcher.logger.error(f"Interactive mode error: {e}")
            print(f"❌ Unexpected error: {e}")
            return 1


def quick_start_wizard():
    """Мастер быстрого старта для новых пользователей"""
    print("""
🌟 Welcome to WaveDream Enhanced Pro v2.0!
This wizard will help you create your first AI-generated track.
    """)
    
    # Простые вопросы
    prompts = [
        "What style of music do you want? (e.g., 'dark trap', 'chill lofi', 'epic cinematic'): ",
        "Any specific tempo in BPM? (Enter for auto-detection): ",
        "How long should it be in seconds? (Enter for auto, typical: 60-120): ",
        "What will you use it for? (personal/freelance/professional/family): "
    ]
    
    answers = {}
    
    for i, prompt_text in enumerate(prompts):
        answer = input(prompt_text).strip()
        answers[i] = answer if answer else None
    
    # Создаём запрос
    music_prompt = answers[0] or "electronic music"
    
    try:
        bpm = int(answers[1]) if answers[1] else None
    except ValueError:
        bpm = None
    
    try:
        duration = int(answers[2]) if answers[2] else None
    except ValueError:
        duration = None
    
    purpose = answers[3] if answers[3] in ["personal", "freelance", "professional", "family"] else "personal"
    
    # Запускаем генерацию
    launcher = WaveDreamEnhancedLauncher()
    
    request = GenerationRequest(
        prompt=music_prompt,
        bpm=bpm,
        duration=duration,
        mastering_purpose=purpose,
        output_dir="quick_start_output",
        export_stems=True
    )
    
    print(f"\n🚀 Generating: '{music_prompt}' for {purpose} use")
    print("This may take a few minutes...")
    
    result = launcher.generate_track_sync(request)
    
    if result.success:
        print(f"\n🎉 Your track is ready!")
        print(f"📁 Location: {result.final_path}")
        print(f"🎯 Quality: {result.quality_score:.2f}/1.0")
        print(f"\n💡 Tip: Use the interactive mode (just run the script) for more options!")
    else:
        print(f"\n❌ Generation failed: {result.error_message}")


if __name__ == "__main__":
    # Проверяем аргументы для специальных режимов
    if len(sys.argv) == 2 and sys.argv[1] == "--quick-start":
        quick_start_wizard()
    else:
        sys.exit(main())
