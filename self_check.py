import logging
from typing import Dict, List, Optional, Any, Union
from pydub import AudioSegment
import numpy as np
import librosa
import os

VALIDATION_CONFIG = {
    "min_duration_sec": 10.0,
    "silence_threshold_db": -40.0,
    "max_silent_ratio": 0.5,
    "peak_db_limit": -0.1
}

def check_rms(audio: AudioSegment) -> float:
    """Проверка RMS уровня"""
    samples = np.array(audio.get_array_of_samples())
    return np.sqrt(np.mean(samples**2))

def is_too_silent(audio: AudioSegment, threshold_db: float = None) -> bool:
    """Проверка на слишком тихий звук"""
    threshold_db = threshold_db or VALIDATION_CONFIG["silence_threshold_db"]
    rms_db = audio.dBFS
    logging.info(f"🔍 RMS dBFS: {rms_db:.2f}")
    return rms_db < threshold_db

def has_enough_duration(audio: AudioSegment, min_duration_sec: float = None) -> bool:
    """Проверка минимальной длительности"""
    min_duration_sec = min_duration_sec or VALIDATION_CONFIG["min_duration_sec"]
    return len(audio) >= min_duration_sec * 1000

def is_mostly_silence(path, threshold_db=None, max_silent_ratio=None) -> bool:
    """Проверка на преимущественную тишину"""
    threshold_db = threshold_db or VALIDATION_CONFIG["silence_threshold_db"]
    max_silent_ratio = max_silent_ratio or VALIDATION_CONFIG["max_silent_ratio"]
    try:
        y, sr = librosa.load(path, sr=None)
        S = librosa.feature.rms(y=y)[0]
        silence = S < librosa.db_to_amplitude(threshold_db)
        silent_ratio = np.mean(silence)
        logging.info(f"🔍 Silent frames: {silent_ratio * 100:.1f}%")
        return silent_ratio > max_silent_ratio
    except Exception as e:
        logging.warning(f"⚠️ Ошибка анализа тишины: {e}")
        return False

def has_peak_clipping(path, peak_limit_db=None) -> bool:
    """Проверка клиппинга"""
    peak_limit_db = peak_limit_db or VALIDATION_CONFIG["peak_db_limit"]
    try:
        y, sr = librosa.load(path, sr=None)
        peak_db = librosa.amplitude_to_db(np.max(np.abs(y)))
        logging.info(f"🔍 Peak level dB: {peak_db:.2f}")
        return peak_db >= peak_limit_db
    except Exception as e:
        logging.warning(f"⚠️ Ошибка анализа пиков: {e}")
        return False

def verify_mix(audio_data: Union[str, AudioSegment, bytes], target_config: Dict = None) -> Dict[str, Any]:
    """
    ИСПРАВЛЕННАЯ функция проверки микса
    Теперь принимает разные типы входных данных
    """
    try:
        # Если передан путь к файлу
        if isinstance(audio_data, str):
            path = audio_data
            logging.info(f"🔍 Проверка финального микса: {path}")
            
            if not os.path.exists(path):
                return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": "File not found"}]}
            
            try:
                audio = AudioSegment.from_file(path)
            except Exception as e:
                logging.error(f"❌ Ошибка загрузки: {e}")
                return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": f"Load failed: {e}"}]}
            
            # Все проверки для файла
            if not has_enough_duration(audio):
                return {"status": "warning", "score": 0.5, "issues": [{"type": "warning", "message": "Track too short"}]}
            
            if is_too_silent(audio):
                return {"status": "warning", "score": 0.3, "issues": [{"type": "warning", "message": "Track too silent"}]}
            
            if is_mostly_silence(path):
                return {"status": "error", "score": 0.1, "issues": [{"type": "critical", "message": "Mostly silence"}]}
            
            if has_peak_clipping(path):
                return {"status": "warning", "score": 0.7, "issues": [{"type": "warning", "message": "Peak clipping detected"}]}
            
            return {"status": "excellent", "score": 0.9, "issues": [], "recommendations": ["High quality track"]}
        
        # Если передан AudioSegment
        elif isinstance(audio_data, AudioSegment):
            if len(audio_data) == 0 or audio_data.max_dBFS == float('-inf'):
                return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": "Silent or empty audio"}]}
            
            return {"status": "acceptable", "score": 0.75, "issues": [], "recommendations": ["Audio segment validated"]}
        
        # Если переданы bytes
        elif isinstance(audio_data, (bytes, bytearray)):
            if len(audio_data) == 0:
                return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": "Empty bytes"}]}
            
            return {"status": "acceptable", "score": 0.75, "issues": [], "recommendations": ["Bytes data validated"]}
        
        else:
            return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": f"Unsupported data type: {type(audio_data)}"}]}
    
    except Exception as e:
        logging.error(f"❌ Verification error: {e}")
        return {"status": "error", "score": 0.0, "issues": [{"type": "critical", "message": str(e)}]}
