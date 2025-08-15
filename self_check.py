import os
import logging
from pydub import AudioSegment
import numpy as np
import librosa

VALIDATION_CONFIG = {
    "min_duration_sec": 10.0,
    "silence_threshold_db": -40.0,
    "max_silent_ratio": 0.5,
    "peak_db_limit": -0.1
}

def check_rms(audio: AudioSegment) -> float:
    samples = np.array(audio.get_array_of_samples())
    return np.sqrt(np.mean(samples**2))

def is_too_silent(audio: AudioSegment, threshold_db: float = None) -> bool:
    threshold_db = threshold_db or VALIDATION_CONFIG["silence_threshold_db"]
    rms_db = audio.dBFS
    logging.info(f"🔍 RMS dBFS: {rms_db:.2f}")
    return rms_db < threshold_db

def has_enough_duration(audio: AudioSegment, min_duration_sec: float = None) -> bool:
    min_duration_sec = min_duration_sec or VALIDATION_CONFIG["min_duration_sec"]
    return len(audio) >= min_duration_sec * 1000

def is_mostly_silence(path, threshold_db=None, max_silent_ratio=None) -> bool:
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
    peak_limit_db = peak_limit_db or VALIDATION_CONFIG["peak_db_limit"]
    try:
        y, sr = librosa.load(path, sr=None)
        peak_db = librosa.amplitude_to_db(np.max(np.abs(y)))
        logging.info(f"🔍 Peak level dB: {peak_db:.2f}")
        return peak_db >= peak_limit_db
    except Exception as e:
        logging.warning(f"⚠️ Ошибка анализа пиков: {e}")
        return False

def verify_mix(path: str) -> dict:
    logging.info(f"🔍 Проверка финального микса: {path}")

    if not os.path.exists(path):
        return {"ok": False, "reason": "no_file"}

    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки: {e}")
        return {"ok": False, "reason": "load_fail"}

    if not has_enough_duration(audio):
        return {"ok": False, "reason": "too_short"}

    if is_too_silent(audio):
        return {"ok": False, "reason": "too_silent"}

    if is_mostly_silence(path):
        return {"ok": False, "reason": "mostly_silent"}

    if has_peak_clipping(path):
        return {"ok": False, "reason": "peak_clipping"}

    return {"ok": True}
