# wavedream/core/config.py - Централизованная конфигурация с валидацией
# === ФИКС PYTORCH ДЛЯ WINDOWS ===
def fix_pytorch_windows():
    """Исправляет проблемы с PyTorch на Windows"""
    try:
        # Устанавливаем переменные окружения для PyTorch
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Добавляем пути к библиотекам
        if sys.platform.startswith('win'):
            # Путь к conda/venv библиотекам
            lib_paths = []
            
            # Если виртуальное окружение активно
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                venv_lib = os.path.join(sys.prefix, 'Library', 'bin')
                if os.path.exists(venv_lib):
                    lib_paths.append(venv_lib)
                
                venv_lib2 = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
                if os.path.exists(venv_lib2):
                    lib_paths.append(venv_lib2)
            
            # Добавляем пути в PATH
            for path in lib_paths:
                if path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
        
        print("✅ PyTorch Windows fix applied")
        return True
        
    except Exception as e:
        print(f"⚠️ PyTorch fix failed: {e}")
        return False

# Применяем фикс
fix_pytorch_windows()

# Теперь безопасный импорт torch
def safe_import_torch():
    """Безопасный импорт PyTorch с fallback"""
    try:
        import torch
        print(f"✅ PyTorch loaded: {torch.__version__}")
        return torch
    except OSError as e:
        if "shm.dll" in str(e):
            print("❌ PyTorch shm.dll error - trying CPU-only mode")
            try:
                # Принудительно используем CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                import torch
                print(f"✅ PyTorch loaded (CPU-only): {torch.__version__}")
                return torch
            except Exception as e2:
                print(f"❌ PyTorch CPU fallback failed: {e2}")
                return None
        else:
            print(f"❌ PyTorch import error: {e}")
            return None
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return None

# Используй вместо обычного импорта
torch = safe_import_torch()
import sys
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

class MasteringPurpose(Enum):
    FREELANCE = "freelance"
    PROFESSIONAL = "professional" 
    PERSONAL = "personal"
    FAMILY = "family"
    STREAMING = "streaming"
    VINYL = "vinyl"

class GenreType(Enum):
    TRAP = "trap"
    PHONK = "phonk"
    LOFI = "lofi"
    AMBIENT = "ambient"
    EDM = "edm"
    DNB = "dnb"
    TECHNO = "techno"
    HOUSE = "house"
    CINEMATIC = "cinematic"
    HYPERPOP = "hyperpop"
    DRILL = "drill"
    JERSEY = "jersey"

@dataclass
class GenreConfig:
    """Конфигурация жанра с валидацией"""
    name: str
    bpm_range: Tuple[int, int]
    core_instruments: List[str]
    optional_instruments: List[str]
    default_tags: List[str]
    mastering_style: str
    energy_range: Tuple[float, float] = (0.3, 0.9)
    darkness_bias: float = 0.0
    vintage_factor: float = 0.0
    spatial_width: float = 1.0
    harmonic_complexity: float = 0.5
    default_structure: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if self.bpm_range[0] >= self.bpm_range[1]:
            raise ValueError(f"Некорректный BPM диапазон для {self.name}")
        if not self.core_instruments:
            raise ValueError(f"Жанр {self.name} должен иметь основные инструменты")

class WaveDreamConfig:
    """Централизованная конфигурация WaveDream Enhanced Pro"""
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.validate_environment()
        
    # === ПУТИ И ДИРЕКТОРИИ ===
    @property
    def DEFAULT_SAMPLE_DIR(self) -> str:
        paths = [
            r"D:\0\шаблоны\Samples for AKAI",
            "samples", 
            "audio_samples",
            os.path.join(os.path.expanduser("~"), "Documents", "Samples"),
            "D:\\Samples",
            "C:\\Samples"
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Создаём дефолтную
        os.makedirs("samples", exist_ok=True)
        return "samples"
    
    DEFAULT_OUTPUT_DIR = "wavedream_output"
    CACHE_DIR = "wavedream_cache"
    MODELS_DIR = "models"
    
    # === ФАЙЛЫ ИНДЕКСОВ ===
    ENHANCED_INDEX_FILE = "enhanced_sample_index.json"
    BACKUP_INDEX_FILE = "backup_sample_index.json"
    SEMANTIC_CACHE_FILE = "semantic_embeddings.pkl"
    MFCC_FEATURES_FILE = "mfcc_features.npz"
    
    # === ЖАНРОВЫЕ КОНФИГУРАЦИИ ===
    @property
    def GENRE_CONFIGS(self) -> Dict[str, GenreConfig]:
        return {
            GenreType.TRAP.value: GenreConfig(
                name="trap",
                bpm_range=(130, 170),
                core_instruments=["kick", "snare", "hihat", "bass", "808"],
                optional_instruments=["bell", "lead", "vocal", "fx", "pad"],
                default_tags=["808", "trap", "dark", "urban", "aggressive"],
                mastering_style="punchy_aggressive",
                energy_range=(0.6, 0.9),
                darkness_bias=0.3,
                spatial_width=1.2,
                harmonic_complexity=0.3,
                default_structure=[
                    {"type": "intro", "duration": 8, "energy": 0.3},
                    {"type": "verse", "duration": 16, "energy": 0.5},
                    {"type": "hook", "duration": 16, "energy": 0.8},
                    {"type": "verse", "duration": 16, "energy": 0.6},
                    {"type": "hook", "duration": 16, "energy": 0.9},
                    {"type": "outro", "duration": 8, "energy": 0.4}
                ]
            ),
            
            GenreType.LOFI.value: GenreConfig(
                name="lofi",
                bpm_range=(60, 80),
                core_instruments=["soft_kick", "snare", "rim", "vinyl_fx"],
                optional_instruments=["piano", "pad", "jazz_guitar", "rain", "vinyl_crackle"],
                default_tags=["lofi", "chill", "vintage", "cozy", "nostalgic"],
                mastering_style="warm_cozy",
                energy_range=(0.2, 0.5),
                vintage_factor=0.7,
                spatial_width=0.8,
                harmonic_complexity=0.7,
                default_structure=[
                    {"type": "intro", "duration": 15, "energy": 0.2},
                    {"type": "verse", "duration": 30, "energy": 0.4},
                    {"type": "bridge", "duration": 20, "energy": 0.3},
                    {"type": "verse", "duration": 30, "energy": 0.5},
                    {"type": "outro", "duration": 15, "energy": 0.2}
                ]
            ),
            
            GenreType.DNB.value: GenreConfig(
                name="dnb",
                bpm_range=(160, 180),
                core_instruments=["kick", "snare_dnb", "break", "reese_bass"],
                optional_instruments=["pad", "lead", "vocal", "fx", "hat"],
                default_tags=["dnb", "neurofunk", "liquid", "breakbeat", "bass"],
                mastering_style="tight_punchy",
                energy_range=(0.7, 1.0),
                spatial_width=1.1,
                harmonic_complexity=0.4,
                default_structure=[
                    {"type": "intro", "duration": 16, "energy": 0.4},
                    {"type": "buildup", "duration": 16, "energy": 0.6},
                    {"type": "drop", "duration": 32, "energy": 0.9},
                    {"type": "breakdown", "duration": 16, "energy": 0.5},
                    {"type": "drop", "duration": 32, "energy": 1.0},
                    {"type": "outro", "duration": 16, "energy": 0.3}
                ]
            ),
            
            GenreType.AMBIENT.value: GenreConfig(
                name="ambient",
                bpm_range=(60, 90),
                core_instruments=["pad", "texture", "drone", "ambient_fx"],
                optional_instruments=["piano", "strings", "field_recording", "bells", "choir"],
                default_tags=["ambient", "ethereal", "spacious", "meditation", "peaceful"],
                mastering_style="spacious_ethereal", 
                energy_range=(0.1, 0.4),
                spatial_width=1.5,
                harmonic_complexity=0.8,
                default_structure=[
                    {"type": "emergence", "duration": 45, "energy": 0.1},
                    {"type": "development", "duration": 90, "energy": 0.3},
                    {"type": "climax", "duration": 60, "energy": 0.4},
                    {"type": "resolution", "duration": 45, "energy": 0.2}
                ]
            ),
            
            # Добавляем остальные жанры...
            GenreType.TECHNO.value: GenreConfig(
                name="techno",
                bpm_range=(120, 135), 
                core_instruments=["kick_techno", "hat", "tech_bass"],
                optional_instruments=["stab", "modular", "fx", "percussion"],
                default_tags=["techno", "minimal", "hypnotic", "industrial", "warehouse"],
                mastering_style="industrial_clean",
                energy_range=(0.6, 0.9),
                spatial_width=1.0,
                harmonic_complexity=0.3
            )
        }
    
    # === СЕМАНТИЧЕСКАЯ КАРТА РАСШИРЕННАЯ ===
    @property
    def SEMANTIC_MAP(self) -> Dict[str, Dict]:
        return {
            "kick": {
                "synonyms": ["kick", "bd", "bass_drum", "thump", "punch", "boom", "sub_kick"],
                "related": ["808", "sub", "low", "fundamental"],
                "frequency_range": (20, 200),
                "energy_contribution": 0.8,
                "genre_variants": {
                    "trap": ["808_kick", "hard_kick", "punchy_kick", "sub_kick"],
                    "techno": ["techno_kick", "industrial_kick", "pounding_kick"],
                    "house": ["house_kick", "four_on_floor", "groove_kick"],
                    "dnb": ["dnb_kick", "tight_kick", "sharp_kick"]
                }
            },
            
            "snare": {
                "synonyms": ["snare", "snr", "crack", "snap", "backbeat"],
                "related": ["clap", "rim", "percussion", "ghost_snare"],
                "frequency_range": (100, 8000),
                "energy_contribution": 0.7,
                "genre_variants": {
                    "trap": ["trap_snare", "snappy_snare", "tight_snare"],
                    "dnb": ["dnb_snare", "rolling_snare", "break_snare"],
                    "lofi": ["soft_snare", "vintage_snare", "muffled_snare"],
                    "techno": ["minimal_snare", "clap", "tech_snare"]
                }
            },
            
            "hihat": {
                "synonyms": ["hat", "hh", "hi_hat", "closed_hat", "open_hat"],
                "related": ["cymbal", "shaker", "percussion", "groove"],
                "frequency_range": (5000, 20000),
                "energy_contribution": 0.4,
                "genre_variants": {
                    "trap": ["trap_hat", "rolling_hat", "drill_hat"],
                    "house": ["house_hat", "swing_hat", "disco_hat"],
                    "techno": ["tech_hat", "minimal_hat", "analog_hat"]
                }
            },
            
            "bass": {
                "synonyms": ["bass", "sub", "low", "bassline", "low_end"],
                "related": ["808", "reese", "wobble", "growl", "fundamental"],
                "frequency_range": (20, 300),
                "energy_contribution": 0.9,
                "genre_variants": {
                    "trap": ["808", "sub_bass", "sliding_bass"],
                    "dnb": ["reese", "neurobass", "liquid_bass", "growl"],
                    "house": ["house_bass", "deep_bass", "groove_bass"],
                    "techno": ["acid_bass", "modular_bass", "rolling_bass"],
                    "ambient": ["sub_drone", "deep_pad", "atmospheric_bass"]
                }
            },
            
            # Мелодические инструменты
            "lead": {
                "synonyms": ["lead", "melody", "synth", "keys", "main"],
                "related": ["arp", "sequence", "hook", "melody"],
                "frequency_range": (200, 15000),
                "energy_contribution": 0.6,
                "genre_variants": {
                    "trap": ["trap_melody", "dark_lead", "minor_lead"],
                    "house": ["piano", "organ", "electric_piano"],
                    "techno": ["acid_lead", "analog_lead", "modular_lead"],
                    "ambient": ["soft_lead", "ethereal_lead", "pad_lead"]
                }
            },
            
            "pad": {
                "synonyms": ["pad", "strings", "ambient", "texture", "atmosphere"],
                "related": ["drone", "wash", "background", "sustained"],
                "frequency_range": (100, 12000),
                "energy_contribution": 0.3,
                "genre_variants": {
                    "ambient": ["ambient_pad", "space_pad", "ethereal_pad"],
                    "house": ["warm_pad", "analog_pad", "string_pad"],
                    "techno": ["dark_pad", "industrial_pad", "minimal_pad"]
                }
            },
            
            "vocal": {
                "synonyms": ["vocal", "voice", "vox", "vocals", "sung"],
                "related": ["choir", "chant", "rap", "spoken", "human"],
                "frequency_range": (80, 12000),
                "energy_contribution": 0.8,
                "genre_variants": {
                    "trap": ["rap", "autotune", "vocal_chop", "melodic_rap"],
                    "house": ["soulful_vocal", "disco_vocal", "diva_vocal"],
                    "ambient": ["ethereal_vocal", "choir", "chant", "whisper"],
                    "dnb": ["liquid_vocal", "chopped_vocal", "processed_vocal"]
                }
            },
            
            "fx": {
                "synonyms": ["fx", "effect", "sfx", "sound_effect", "transition"],
                "related": ["sweep", "riser", "impact", "whoosh", "noise"],
                "frequency_range": (20, 20000),
                "energy_contribution": 0.2,
                "genre_variants": {
                    "trap": ["trap_fx", "reverse_fx", "vinyl_stop", "air_horn"],
                    "house": ["filter_sweep", "vocal_fx", "piano_fx"],
                    "techno": ["industrial_fx", "noise_fx", "modular_fx"],
                    "ambient": ["nature_fx", "space_fx", "field_recording"]
                }
            }
        }
    
    # === ПАРАМЕТРЫ АНАЛИЗА АУДИО ===
    AUDIO_ANALYSIS = {
        "max_duration": 15,  # увеличили для лучшего анализа
        "sample_rate": 22050,
        "hop_length": 512,
        "frame_size": 2048,
        "n_mels": 128,
        "n_mfcc": 13,
        "chroma_bins": 12,
        "spectral_features": ["centroid", "rolloff", "zero_crossing_rate", "bandwidth"],
        "onset_detection": True,
        "beat_tracking": True,
        "key_detection": True,
        "tempo_estimation": True
    }
    
    # === ПАРАМЕТРЫ ПОДБОРА СЭМПЛОВ ===
    SAMPLE_MATCHING = {
        "min_score_threshold": 3,  # снизили для большей гибкости
        "tempo_tolerance": 25,  # увеличили диапазон
        "semantic_weight": 0.4,
        "tempo_weight": 0.15,
        "genre_weight": 0.15,
        "energy_weight": 0.1,
        "spectral_weight": 0.1,
        "quality_weight": 0.1,
        "use_fuzzy_matching": True,
        "fuzzy_threshold": 0.65,
        "enable_semantic_embeddings": True,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # === КОНФИГИ МАСТЕРИНГА ===
    @property 
    def MASTERING_CONFIGS(self) -> Dict[str, Dict]:
        return {
            MasteringPurpose.FREELANCE.value: {
                "target_lufs": -14,
                "peak_ceiling": -1,
                "dynamic_range": 8,
                "eq_curve": "commercial_bright",
                "compression_style": "punchy",
                "stereo_enhancement": 1.15,
                "harmonic_saturation": 0.3,
                "creative_fx": ["tape_saturation", "vintage_compressor", "stereo_imaging"],
                "character": "punchy commercial sound optimized for streaming"
            },
            
            MasteringPurpose.PROFESSIONAL.value: {
                "target_lufs": -23,
                "peak_ceiling": -3,
                "dynamic_range": 12,
                "eq_curve": "broadcast_standard",
                "compression_style": "transparent",
                "stereo_enhancement": 1.05,
                "harmonic_saturation": 0.2,
                "creative_fx": ["analog_console", "broadcast_limiter"],
                "character": "professional broadcast-ready with full dynamic range"
            },
            
            MasteringPurpose.PERSONAL.value: {
                "target_lufs": -16,
                "peak_ceiling": -2,
                "dynamic_range": 10,
                "eq_curve": "neutral_smooth",
                "compression_style": "gentle",
                "stereo_enhancement": 1.0,
                "harmonic_saturation": 0.15,
                "creative_fx": ["gentle_compression"],
                "character": "clean natural sound for personal listening"
            },
            
            MasteringPurpose.STREAMING.value: {
                "target_lufs": -14,
                "peak_ceiling": -1,
                "dynamic_range": 7,
                "eq_curve": "streaming_optimized", 
                "compression_style": "modern",
                "stereo_enhancement": 1.1,
                "harmonic_saturation": 0.25,
                "creative_fx": ["multiband_compressor", "peak_limiter"],
                "character": "optimized for streaming platforms and loudness normalization"
            },
            
            MasteringPurpose.VINYL.value: {
                "target_lufs": -18,
                "peak_ceiling": -6,
                "dynamic_range": 14,
                "eq_curve": "vinyl_compatible",
                "compression_style": "vintage",
                "stereo_enhancement": 0.9,
                "harmonic_saturation": 0.4,
                "creative_fx": ["vintage_eq", "tape_compression"],
                "character": "warm analog sound optimized for vinyl pressing"
            }
        }
    
    # === ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ ===
    PERFORMANCE = {
        "batch_size": 50,
        "max_workers": 6,  # увеличили
        "cache_size": 2000,  # увеличили кэш
        "index_rebuild_threshold": 0.15,
        "enable_multiprocessing": True,
        "memory_limit_mb": 4096,
        "enable_gpu_acceleration": True,
        "chunk_processing": True,
        "async_sample_loading": True
    }
    
    # === КАЧЕСТВЕННЫЕ ФИЛЬТРЫ ===
    QUALITY_FILTERS = {
        "min_duration": 0.3,
        "max_duration": 600,  # 10 минут
        "min_sample_rate": 44100,
        "max_sample_rate": 192000,
        "min_bit_depth": 16,
        "exclude_extensions": [".tmp", ".bak", ".old", ".log"],
        "exclude_folders": ["backup", "temp", "trash", "old", "cache"],
        "min_file_size_kb": 10,
        "max_file_size_mb": 100,
        "audio_quality_threshold": 0.7,
        "silence_threshold": 0.01,
        "max_silence_ratio": 0.8
    }
    
    # === НАСТРОЙКИ ЛОГИРОВАНИЯ ===
    LOGGING = {
        "level": "INFO",
        "format": "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        "file": "wavedream_enhanced.log",
        "max_size_mb": 100,
        "backup_count": 5,
        "enable_debug": False,
        "log_performance": True,
        "log_sample_matches": True
    }
    
    # === ВАЛИДАЦИЯ КОНФИГУРАЦИИ ===
    def validate_environment(self) -> List[str]:
        """Полная валидация окружения"""
        errors = []
        warnings = []
        
        # Проверка директорий
        sample_dir = self.DEFAULT_SAMPLE_DIR
        if not os.path.exists(sample_dir):
            errors.append(f"Sample directory not found: {sample_dir}")
        
        # Проверка зависимостей
        required_packages = [
            "torch", "torchaudio", "librosa", "pydub", "soundfile", 
            "numpy", "scipy", "sklearn", "sentence_transformers"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                errors.append(f"Required package not found: {package}")
        
        # Проверка параметров
        if self.PERFORMANCE["max_workers"] < 1:
            errors.append("max_workers must be >= 1")
        
        if self.AUDIO_ANALYSIS["max_duration"] < 1:
            errors.append("max_duration must be >= 1")
        
        # Проверка жанровых конфигураций
        for genre_name, config in self.GENRE_CONFIGS.items():
            try:
                # Валидация через __post_init__
                pass  
            except ValueError as e:
                errors.append(f"Genre config error for {genre_name}: {e}")
        
        # Проверка GPU
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("CUDA not available, using CPU (slower)")
        except ImportError:
            pass
        
        # Проверка памяти
        import psutil
        available_ram = psutil.virtual_memory().available / (1024**3)  # GB
        if available_ram < 4:
            warnings.append(f"Low RAM detected: {available_ram:.1f}GB (recommended: 8GB+)")
        
        if errors:
            raise RuntimeError(f"Configuration errors: {'; '.join(errors)}")
        
        if warnings:
            logging.warning(f"Configuration warnings: {'; '.join(warnings)}")
        
        return errors + warnings
    
    def get_genre_config(self, genre: str) -> Optional[GenreConfig]:
        """Получение конфига жанра с fallback"""
        return self.GENRE_CONFIGS.get(genre.lower(), self.GENRE_CONFIGS.get("trap"))
    
    def get_mastering_config(self, purpose: Union[str, MasteringPurpose]) -> Dict:
        """Получение конфига мастеринга с fallback"""
        key = purpose.value if isinstance(purpose, MasteringPurpose) else str(purpose)
        return self.MASTERING_CONFIGS.get(key, self.MASTERING_CONFIGS[MasteringPurpose.PERSONAL.value])
    
    def export_config(self, path: str) -> None:
        """Экспорт конфигурации в JSON"""
        config_dict = {
            "version": "2.0.0",
            "genres": {name: {
                "name": config.name,
                "bpm_range": config.bpm_range,
                "core_instruments": config.core_instruments,
                "optional_instruments": config.optional_instruments,
                "mastering_style": config.mastering_style
            } for name, config in self.GENRE_CONFIGS.items()},
            "mastering_purposes": list(self.MASTERING_CONFIGS.keys()),
            "audio_analysis": self.AUDIO_ANALYSIS,
            "sample_matching": self.SAMPLE_MATCHING,
            "performance": self.PERFORMANCE
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# Создаём глобальный экземпляр конфига
config = WaveDreamConfig()
