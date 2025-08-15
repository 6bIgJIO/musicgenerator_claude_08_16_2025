# wavedream/core/verification.py - Система верификации качества

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from io import BytesIO

# Аудио анализ
from scipy.signal import find_peaks 
from pydub import AudioSegment
import librosa
import soundfile as sf


@dataclass
class QualityIssue:
    """Описание проблемы качества"""
    severity: str  # "critical", "warning", "info"
    category: str  # "loudness", "dynamics", "spectrum", "stereo", "artifacts"
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    suggestion: Optional[str] = None


class MixVerifier:
    """
    Система верификации качества финального микса
    
    Проверяет:
    - Соответствие стандартам громкости
    - Динамический диапазон
    - Спектральный баланс
    - Стерео характеристики
    - Артефакты обработки
    - Клиппинг и искажения
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Пороги для разных стандартов
        self.standards = {
            "streaming": {"lufs_range": (-16, -12), "peak_max": -1, "dr_min": 6},
            "broadcast": {"lufs_range": (-25, -20), "peak_max": -3, "dr_min": 10},
            "cd": {"lufs_range": (-18, -8), "peak_max": -0.1, "dr_min": 8},
            "vinyl": {"lufs_range": (-20, -15), "peak_max": -6, "dr_min": 12}
        }

    def _safe_audio_conversion(self, audio: Union[bytes, AudioSegment, str]) -> AudioSegment:
        """
        ИСПРАВЛЕНО: Безопасная конвертация аудио без потери данных
        """
        try:
            if isinstance(audio, AudioSegment):
                return audio
            elif isinstance(audio, bytes):
                # ИСПРАВЛЕНИЕ: Правильная конвертация из bytes
                self.logger.info("Converting bytes to AudioSegment...")
                return AudioSegment.from_file(BytesIO(audio))
            elif isinstance(audio, str):
                # Путь к файлу
                self.logger.info(f"Loading audio from file: {audio}")
                return AudioSegment.from_file(audio)
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
                
        except Exception as e:
            self.logger.error(f"❌ Audio conversion failed: {e}")
            raise RuntimeError(f"Failed to convert audio: {e}")
    
    def _normalize_samples(self, samples: np.ndarray, sample_width: int) -> np.ndarray:
        """
        ИСПРАВЛЕНО: Правильная нормализация для разных битовых глубин
        """
        if sample_width == 1:  # 8-bit
            return samples.astype(np.float32) / 128.0
        elif sample_width == 2:  # 16-bit
            return samples.astype(np.float32) / 32768.0
        elif sample_width == 3:  # 24-bit
            return samples.astype(np.float32) / 8388608.0
        elif sample_width == 4:  # 32-bit float или int
            if samples.dtype == np.float32:
                return samples  # Уже нормализованы
            else:
                return samples.astype(np.float32) / 2147483648.0
        else:
            # Автоматическое определение
            max_val = np.max(np.abs(samples))
            return samples.astype(np.float32) / max_val
    
    async def analyze_track(self, audio: Union[bytes, AudioSegment, str], target_config: Dict) -> Dict:
        """
        ИСПРАВЛЕНО: Полный анализ качества трека без потери данных
        """
        self.logger.info("🔍 Starting quality verification...")
        
        try:
            # ИСПРАВЛЕНИЕ: Безопасная конвертация
            audio_segment = self._safe_audio_conversion(audio)
            
            # Проверяем, что аудио не пустое
            if len(audio_segment) == 0:
                raise ValueError("Empty audio segment")
            
            self.logger.info(f"  📊 Audio info: {len(audio_segment)}ms, {audio_segment.channels}ch, {audio_segment.frame_rate}Hz")
            
            # Проводим все проверки
            issues = []
            metrics = {}
            
            # 1. Анализ громкости
            loudness_issues, loudness_metrics = await self._check_loudness(audio_segment, target_config)
            issues.extend(loudness_issues)
            metrics.update(loudness_metrics)
            
            # 2. Анализ динамики
            dynamics_issues, dynamics_metrics = await self._check_dynamics(audio_segment, target_config)
            issues.extend(dynamics_issues)
            metrics.update(dynamics_metrics)
            
            # 3. Спектральный анализ
            spectrum_issues, spectrum_metrics = await self._check_spectrum(audio_segment, target_config)
            issues.extend(spectrum_issues)
            metrics.update(spectrum_metrics)
            
            # 4. Стерео анализ
            stereo_issues, stereo_metrics = await self._check_stereo(audio_segment, target_config)
            issues.extend(stereo_issues)
            metrics.update(stereo_metrics)
            
            # 5. Проверка артефактов
            artifacts_issues, artifacts_metrics = await self._check_artifacts(audio_segment)
            issues.extend(artifacts_issues)
            metrics.update(artifacts_metrics)
            
            # Составляем общий отчёт
            report = self._compile_quality_report(issues, metrics, target_config)
            
            # Логируем результат
            overall_score = report["overall_score"]
            critical_count = len([i for i in issues if i.severity == "critical"])
            warning_count = len([i for i in issues if i.severity == "warning"])
            
            self.logger.info(f"  ✅ Quality analysis complete: {overall_score:.2f}/1.0")
            if critical_count > 0:
                self.logger.warning(f"  ⚠️ {critical_count} critical issues found")
            if warning_count > 0:
                self.logger.info(f"  ℹ️ {warning_count} warnings")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Quality analysis error: {e}")
            return {
                "success": False,
                "overall_score": 0.0,
                "error": str(e),
                "issues": [QualityIssue(
                    severity="critical",
                    category="system",
                    message=f"Analysis failed: {str(e)}",
                    suggestion="Check audio format and file integrity"
                ).__dict__],
                "metrics": {}
            }
    
    async def _check_loudness(self, audio: AudioSegment, target_config: Dict) -> Tuple[List[QualityIssue], Dict]:
        """Проверка параметров громкости"""
        issues = []
        metrics = {}
        
        try:
            # Базовые измерения
            peak_db = audio.max_dBFS
            rms = audio.rms
            
            # Приближенный LUFS (в реальности нужен pyloudnorm)
            lufs_approx = 10 * np.log10(rms**2 / (32768**2)) - 0.691
            lufs_approx = max(-70, lufs_approx)
            
            metrics.update({
                "peak_db": peak_db,
                "rms": rms,
                "lufs_integrated": lufs_approx
            })
            
            # Проверяем соответствие целям
            target_lufs = target_config.get("target_lufs", -14)
            target_peak = target_config.get("peak_ceiling", -1)
            
            # LUFS проверка
            lufs_error = abs(lufs_approx - target_lufs)
            if lufs_error > 2.0:
                issues.append(QualityIssue(
                    severity="critical",
                    category="loudness",
                    message=f"LUFS deviation too high: {lufs_approx:.1f} vs target {target_lufs:.1f}",
                    value=lufs_approx,
                    threshold=target_lufs,
                    suggestion="Adjust overall level or mastering limiter settings"
                ))
            elif lufs_error > 1.0:
                issues.append(QualityIssue(
                    severity="warning",
                    category="loudness", 
                    message=f"LUFS slightly off target: {lufs_approx:.1f} vs {target_lufs:.1f}",
                    value=lufs_approx,
                    threshold=target_lufs
                ))
            
            # Peak проверка
            if peak_db > target_peak:
                issues.append(QualityIssue(
                    severity="critical",
                    category="loudness",
                    message=f"Peak level too high: {peak_db:.1f}dB vs ceiling {target_peak:.1f}dB",
                    value=peak_db,
                    threshold=target_peak,
                    suggestion="Apply peak limiting"
                ))
            
            # Проверка на возможный клиппинг
            if peak_db > -0.1:
                issues.append(QualityIssue(
                    severity="critical",
                    category="loudness",
                    message="Possible digital clipping detected",
                    value=peak_db,
                    suggestion="Reduce overall level to prevent clipping"
                ))
            
        except Exception as e:
            self.logger.error(f"Loudness check error: {e}")
        
        return issues, metrics
    
    async def _check_dynamics(self, audio: AudioSegment, target_config: Dict) -> Tuple[List[QualityIssue], Dict]:
        """Проверка динамических характеристик"""
        issues = []
        metrics = {}
        
        try:
            # Приближенный расчёт динамического диапазона
            peak_db = audio.max_dBFS
            rms_db = 20 * np.log10(audio.rms / 32768)
            
            dynamic_range = peak_db - rms_db
            metrics["dynamic_range"] = dynamic_range
            
            # Целевой динамический диапазон
            target_dr = target_config.get("dynamic_range", 8)
            
            if dynamic_range < target_dr - 3:
                issues.append(QualityIssue(
                    severity="warning",
                    category="dynamics",
                    message=f"Low dynamic range: {dynamic_range:.1f}LU vs target {target_dr:.1f}LU",
                    value=dynamic_range,
                    threshold=target_dr,
                    suggestion="Reduce compression or use parallel compression"
                ))
            elif dynamic_range > target_dr + 5:
                issues.append(QualityIssue(
                    severity="info",
                    category="dynamics",
                    message=f"High dynamic range: {dynamic_range:.1f}LU (may be good for some genres)",
                    value=dynamic_range
                ))
            
            # Проверка на чрезмерную компрессию
            if dynamic_range < 4:
                issues.append(QualityIssue(
                    severity="critical",
                    category="dynamics",
                    message="Excessive compression detected - very low dynamic range",
                    value=dynamic_range,
                    suggestion="Reduce compression ratio or increase attack time"
                ))
            
        except Exception as e:
            self.logger.error(f"Dynamics check error: {e}")
        
        return issues, metrics
    
    async def _check_spectrum(self, audio: AudioSegment, target_config: Dict) -> Tuple[List[QualityIssue], Dict]:
        """ИСПРАВЛЕНО: Проверка спектрального баланса с сохранением стерео"""
        issues = []
        metrics = {}
        
        try:
            # ИСПРАВЛЕНИЕ: Правильная обработка стерео/моно
            samples = np.array(audio.get_array_of_samples())
            sample_width = audio.sample_width
            
            # Нормализация с учётом битовой глубины
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Сохраняем информацию о стерео для анализа баланса
                left_channel = self._normalize_samples(samples[:, 0], sample_width)
                right_channel = self._normalize_samples(samples[:, 1], sample_width)
                # Для спектрального анализа используем сумму каналов, а не среднее
                mono_samples = (left_channel + right_channel) / 2.0
                metrics["stereo_balance"] = {
                    "left_rms": float(np.sqrt(np.mean(left_channel**2))),
                    "right_rms": float(np.sqrt(np.mean(right_channel**2)))
                }
            else:
                mono_samples = self._normalize_samples(samples, sample_width)
            
            sample_rate = audio.frame_rate
            
            # FFT анализ
            fft = np.fft.rfft(mono_samples)
            freqs = np.fft.rfftfreq(len(mono_samples), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Частотные полосы (расширенные)
            sub_low_mask = freqs < 80        # Суб-басы
            low_mask = (freqs >= 80) & (freqs < 300)      # Басы
            low_mid_mask = (freqs >= 300) & (freqs < 1000) # Низкие средние
            mid_mask = (freqs >= 1000) & (freqs < 3000)    # Средние
            high_mid_mask = (freqs >= 3000) & (freqs < 8000) # Высокие средние
            high_mask = freqs >= 8000        # Высокие
            
            total_energy = np.sum(magnitude**2)
            if total_energy > 0:  # Избегаем деления на ноль
                sub_low_energy = np.sum(magnitude[sub_low_mask]**2) / total_energy
                low_energy = np.sum(magnitude[low_mask]**2) / total_energy
                low_mid_energy = np.sum(magnitude[low_mid_mask]**2) / total_energy
                mid_energy = np.sum(magnitude[mid_mask]**2) / total_energy
                high_mid_energy = np.sum(magnitude[high_mid_mask]**2) / total_energy
                high_energy = np.sum(magnitude[high_mask]**2) / total_energy
                
                metrics.update({
                    "frequency_balance": {
                        "sub_low": float(sub_low_energy),
                        "low": float(low_energy),
                        "low_mid": float(low_mid_energy),
                        "mid": float(mid_energy),
                        "high_mid": float(high_mid_energy),
                        "high": float(high_energy)
                    }
                })
                
                # Спектральный центроид с защитой от ошибок
                magnitude_sum = np.sum(magnitude)
                if magnitude_sum > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / magnitude_sum
                    metrics["spectral_centroid"] = float(spectral_centroid)
                else:
                    spectral_centroid = 0
                    metrics["spectral_centroid"] = 0.0
                
                # Улучшенные проверки баланса
                total_low = sub_low_energy + low_energy
                total_high = high_mid_energy + high_energy
                
                if total_low > 0.6:
                    issues.append(QualityIssue(
                        severity="warning",
                        category="spectrum",
                        message=f"Very bass-heavy mix ({total_low*100:.1f}%) - may sound muddy",
                        value=total_low,
                        suggestion="Apply high-pass filtering or reduce low frequencies"
                    ))
                
                if total_high > 0.4:
                    issues.append(QualityIssue(
                        severity="warning",
                        category="spectrum", 
                        message=f"Very bright mix ({total_high*100:.1f}%) - may sound harsh",
                        value=total_high,
                        suggestion="Apply gentle high-frequency roll-off"
                    ))
                
                if mid_energy < 0.2:
                    issues.append(QualityIssue(
                        severity="warning",
                        category="spectrum",
                        message="Low midrange content - vocals/instruments may lack presence",
                        value=mid_energy,
                        suggestion="Boost midrange frequencies (1-3kHz)"
                    ))
                
                # Проверка спектрального центроида
                if spectral_centroid > 5000:
                    issues.append(QualityIssue(
                        severity="warning",
                        category="spectrum",
                        message=f"Very bright mix - spectral centroid {spectral_centroid:.0f}Hz",
                        value=spectral_centroid,
                        suggestion="Consider gentle high-frequency roll-off"
                    ))
                elif spectral_centroid < 800 and spectral_centroid > 0:
                    issues.append(QualityIssue(
                        severity="warning",
                        category="spectrum",
                        message=f"Very dark mix - spectral centroid {spectral_centroid:.0f}Hz", 
                        value=spectral_centroid,
                        suggestion="Add high-frequency content or reduce low-mid buildup"
                    ))
            
        except Exception as e:
            self.logger.error(f"Spectrum check error: {e}")
            issues.append(QualityIssue(
                severity="warning",
                category="spectrum",
                message=f"Spectrum analysis failed: {str(e)}",
                suggestion="Check audio file integrity"
            ))
        
        return issues, metrics
    
    async def _check_stereo(self, audio: AudioSegment, target_config: Dict) -> Tuple[List[QualityIssue], Dict]:
        """Проверка стерео характеристик"""
        issues = []
        metrics = {}
        
        try:
            if audio.channels < 2:
                issues.append(QualityIssue(
                    severity="info",
                    category="stereo",
                    message="Mono audio - no stereo imaging",
                    suggestion="Consider stereo enhancement if appropriate"
                ))
                metrics["stereo_width"] = 0.0
                metrics["phase_correlation"] = 1.0
                return issues, metrics
            
            # Стерео анализ
            samples = np.array(audio.get_array_of_samples())
            samples = samples.reshape((-1, 2))
            
            left = samples[:, 0]
            right = samples[:, 1]
            
            # Корреляция между каналами
            correlation = np.corrcoef(left, right)[0, 1]
            if np.isnan(correlation):
                correlation = 1.0
            
            # Стерео ширина
            stereo_width = 1.0 - abs(correlation)
            
            metrics.update({
                "phase_correlation": float(correlation),
                "stereo_width": float(stereo_width)
            })
            
            # Проверки
            if correlation < -0.5:
                issues.append(QualityIssue(
                    severity="critical",
                    category="stereo",
                    message="Strong phase correlation issues - mono compatibility problems",
                    value=correlation,
                    suggestion="Check stereo processing and avoid excessive widening"
                ))
            elif correlation < 0.0:
                issues.append(QualityIssue(
                    severity="warning",
                    category="stereo",
                    message="Some phase correlation issues detected",
                    value=correlation
                ))
            
            if stereo_width < 0.1:
                issues.append(QualityIssue(
                    severity="info", 
                    category="stereo",
                    message="Very narrow stereo image - consider stereo enhancement",
                    value=stereo_width
                ))
            elif stereo_width > 0.9:
                issues.append(QualityIssue(
                    severity="warning",
                    category="stereo",
                    message="Very wide stereo image - check mono compatibility",
                    value=stereo_width,
                    suggestion="Test mono playback compatibility"
                ))
            
        except Exception as e:
            self.logger.error(f"Stereo check error: {e}")
        
        return issues, metrics
    
    async def _check_artifacts(self, audio: AudioSegment) -> Tuple[List[QualityIssue], Dict]:
        """ИСПРАВЛЕНО: Проверка на артефакты с правильной нормализацией"""
        issues = []
        metrics = {}
        
        try:
            # ИСПРАВЛЕНИЕ: Правильная обработка разных форматов
            samples = np.array(audio.get_array_of_samples())
            sample_width = audio.sample_width
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                mono_samples = samples.mean(axis=1)
            else:
                mono_samples = samples
            
            # ИСПРАВЛЕНИЕ: Правильная нормализация
            mono_samples = self._normalize_samples(mono_samples, sample_width)
            sample_rate = audio.frame_rate
            
            # 1. Проверка на клиппинг (адаптивная)
            clipping_threshold = 0.98  # Чуть меньше максимума для учёта погрешностей
            clipping_ratio = np.sum(np.abs(mono_samples) >= clipping_threshold) / len(mono_samples)
            metrics["clipping_ratio"] = float(clipping_ratio)
            
            if clipping_ratio > 0.001:  # Более 0.1% клиппинга
                issues.append(QualityIssue(
                    severity="critical",
                    category="artifacts",
                    message=f"Digital clipping detected: {clipping_ratio*100:.3f}% of samples",
                    value=clipping_ratio,
                    suggestion="Reduce input level or apply proper limiting"
                ))
            elif clipping_ratio > 0.0001:
                issues.append(QualityIssue(
                    severity="warning",
                    category="artifacts",
                    message=f"Minor clipping detected: {clipping_ratio*100:.4f}% of samples",
                    value=clipping_ratio,
                    suggestion="Monitor levels more carefully"
                ))
            
            # 2. ИСПРАВЛЕНО: Проверка на DC offset
            dc_offset = np.mean(mono_samples)
            metrics["dc_offset"] = float(abs(dc_offset))
            
            dc_threshold = 0.01 if sample_width <= 2 else 0.001  # Адаптивный порог
            if abs(dc_offset) > dc_threshold:
                issues.append(QualityIssue(
                    severity="warning",
                    category="artifacts",
                    message=f"DC offset detected: {dc_offset:.4f}",
                    value=abs(dc_offset),
                    suggestion="Apply DC removal filter or check recording chain"
                ))
            
            # 3. Улучшенная проверка на тишину
            if len(mono_samples) > sample_rate * 0.2:  # Минимум 200ms
                window_size = min(int(0.1 * sample_rate), len(mono_samples) // 10)
                
                start_samples = mono_samples[:window_size]
                end_samples = mono_samples[-window_size:]
                
                start_rms = np.sqrt(np.mean(start_samples**2))
                end_rms = np.sqrt(np.mean(end_samples**2))
                
                metrics.update({
                    "start_silence_level": float(start_rms),
                    "end_silence_level": float(end_rms)
                })
                
                # Адаптивный порог тишины в зависимости от общего уровня
                overall_rms = np.sqrt(np.mean(mono_samples**2))
                silence_threshold = max(0.0001, overall_rms * 0.001)  # -60dB от среднего уровня
                
                if start_rms < silence_threshold:
                    issues.append(QualityIssue(
                        severity="info",
                        category="artifacts",
                        message=f"Silent start detected ({20*np.log10(start_rms+1e-10):.1f}dBFS)",
                        value=start_rms,
                        suggestion="Consider trimming silent beginning"
                    ))
                
                if end_rms < silence_threshold:
                    issues.append(QualityIssue(
                        severity="info",
                        category="artifacts", 
                        message=f"Silent end detected ({20*np.log10(end_rms+1e-10):.1f}dBFS)",
                        value=end_rms,
                        suggestion="Consider trimming silent ending"
                    ))
            
            # 4. ИСПРАВЛЕНО: Проверка на чрезмерную компрессию
            if len(mono_samples) > sample_rate * 2:  # Минимум 2 секунды
                chunk_size = int(0.05 * sample_rate)  # 50ms чанки для лучшей точности
                rms_history = []
                
                for i in range(0, len(mono_samples) - chunk_size, chunk_size//2):  # 50% перекрытие
                    chunk = mono_samples[i:i+chunk_size]
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:  # Избегаем логарифма от нуля
                        rms_history.append(chunk_rms)
                
                if len(rms_history) > 20:  # Достаточно данных
                    rms_variation = np.std(rms_history) / np.mean(rms_history)  # Нормализованная вариация
                    metrics["rms_variation"] = float(rms_variation)
                    
                    # Если RMS очень стабильная относительно среднего уровня
                    if rms_variation < 0.1:  # Менее 10% вариации
                        issues.append(QualityIssue(
                            severity="warning",
                            category="artifacts",
                            message=f"Very stable RMS (variation: {rms_variation:.3f}) - possible over-compression",
                            value=rms_variation,
                            suggestion="Check compression settings and preserve dynamics"
                        ))
            
            # 5. ИСПРАВЛЕНО: Проверка THD (Total Harmonic Distortion)
            try:
                # Простая проверка на искажения через анализ гармоник
                fft = np.fft.rfft(mono_samples)
                magnitude = np.abs(fft)
                
                # Ищем пики (требует scipy, но с fallback)
                try:
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.05, distance=10)
                    metrics["spectral_peaks_count"] = len(peaks)
                    
                    # Анализ гармонических искажений
                    if len(peaks) > 0:
                        # Основная частота (самый мощный пик)
                        fundamental_idx = peaks[np.argmax(magnitude[peaks])]
                        fundamental_freq = fundamental_idx * audio.frame_rate / (2 * len(magnitude))
                        
                        # Ищем гармоники
                        harmonics = []
                        for harmonic in range(2, 6):  # 2-я до 5-й гармоники
                            expected_freq = fundamental_freq * harmonic
                            freq_tolerance = 50  # Hz
                            
                            for peak_idx in peaks:
                                peak_freq = peak_idx * audio.frame_rate / (2 * len(magnitude))
                                if abs(peak_freq - expected_freq) < freq_tolerance:
                                    harmonic_level = magnitude[peak_idx] / magnitude[fundamental_idx]
                                    harmonics.append(harmonic_level)
                                    break
                        
                        if harmonics:
                            thd = np.sqrt(np.sum(np.array(harmonics)**2))
                            metrics["thd_estimate"] = float(thd)
                            
                            if thd > 0.1:  # 10% THD
                                issues.append(QualityIssue(
                                    severity="warning",
                                    category="artifacts",
                                    message=f"High harmonic distortion estimated: {thd*100:.1f}%",
                                    value=thd,
                                    suggestion="Check for over-processing or analog distortion"
                                ))
                                
                except ImportError:
                    # Fallback без scipy
                    magnitude_mean = np.mean(magnitude)
                    magnitude_peaks = magnitude[magnitude > magnitude_mean * 3]
                    metrics["spectral_peaks_count"] = len(magnitude_peaks)
                    
                    if len(magnitude_peaks) > len(mono_samples) // 2000:
                        issues.append(QualityIssue(
                            severity="info",
                            category="artifacts",
                            message="High spectral complexity - possible distortion or rich harmonic content",
                            suggestion="Verify if distortion is intentional"
                        ))
                        
            except Exception as fft_error:
                self.logger.warning(f"FFT analysis failed: {fft_error}")
            
        except Exception as e:
            self.logger.error(f"Artifacts check error: {e}")
            issues.append(QualityIssue(
                severity="warning", 
                category="artifacts",
                message=f"Artifacts analysis failed: {str(e)}",
                suggestion="Check audio file integrity"
            ))
        
        return issues, metrics
    
    def _compile_quality_report(self, issues: List[QualityIssue], metrics: Dict, target_config: Dict) -> Dict:
        """Компиляция итогового отчёта о качестве"""
        
        # Подсчитываем проблемы по серьёзности
        critical_issues = [i for i in issues if i.severity == "critical"]
        warning_issues = [i for i in issues if i.severity == "warning"] 
        info_issues = [i for i in issues if i.severity == "info"]
        
        # Расчёт общего скора (0.0 - 1.0)
        base_score = 1.0
        
        # Штрафы за проблемы
        base_score -= len(critical_issues) * 0.2  # Критические проблемы -20%
        base_score -= len(warning_issues) * 0.05   # Предупреждения -5%
        base_score -= len(info_issues) * 0.01      # Информация -1%
        
        overall_score = max(0.0, base_score)
        
        # Определяем общий статус
        if len(critical_issues) > 0:
            status = "critical_issues"
            recommendation = "Critical issues found - manual review required"
        elif len(warning_issues) > 0:
            status = "warnings"
            recommendation = "Some issues detected - review recommended"
        elif len(info_issues) > 0:
            status = "minor_issues"
            recommendation = "Minor issues noted - track acceptable"
        else:
            status = "excellent"
            recommendation = "No issues detected - excellent quality"
        
        # Создаём отчёт
        report = {
            "success": True,
            "overall_score": overall_score,
            "status": status,
            "recommendation": recommendation,
            "summary": {
                "critical_issues": len(critical_issues),
                "warnings": len(warning_issues), 
                "info_issues": len(info_issues),
                "total_issues": len(issues)
            },
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "value": issue.value,
                    "threshold": issue.threshold,
                    "suggestion": issue.suggestion
                }
                for issue in issues
            ],
            "metrics": metrics,
            "target_config": target_config,
            "analysis_timestamp": None,  # В реальности - текущее время
            "compliance": self._check_standards_compliance(metrics, target_config)
        }
        
        return report
    
    def _check_standards_compliance(self, metrics: Dict, target_config: Dict) -> Dict:
        """Проверка соответствия различным стандартам"""
        compliance = {}
        
        current_lufs = metrics.get("lufs_integrated", -14)
        current_peak = metrics.get("peak_db", -1)
        current_dr = metrics.get("dynamic_range", 8)
        
        for standard_name, standard in self.standards.items():
            lufs_range = standard["lufs_range"]
            peak_max = standard["peak_max"]
            dr_min = standard["dr_min"]
            
            # Проверяем соответствие
            lufs_ok = lufs_range[0] <= current_lufs <= lufs_range[1]
            peak_ok = current_peak <= peak_max
            dr_ok = current_dr >= dr_min
            
            compliance[standard_name] = {
                "overall": lufs_ok and peak_ok and dr_ok,
                "loudness": lufs_ok,
                "peaks": peak_ok,
                "dynamics": dr_ok,
                "details": {
                    "lufs_target": lufs_range,
                    "lufs_actual": current_lufs,
                    "peak_max": peak_max,
                    "peak_actual": current_peak,
                    "dr_min": dr_min,
                    "dr_actual": current_dr
                }
            }
        
        return compliance
    
    def generate_markdown_report(self, report: Dict, output_path: str) -> bool:
        """Генерация подробного отчёта в формате Markdown"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 🔍 WaveDream Quality Analysis Report\n\n")
                
                # Общая информация
                f.write(f"## 📊 Overall Assessment\n\n")
                f.write(f"- **Quality Score**: {report['overall_score']:.2f}/1.0\n")
                f.write(f"- **Status**: {report['status'].replace('_', ' ').title()}\n")
                f.write(f"- **Recommendation**: {report['recommendation']}\n\n")
                
                # Сводка проблем
                summary = report['summary']
                f.write(f"## 🚨 Issues Summary\n\n")
                f.write(f"- **Critical Issues**: {summary['critical_issues']}\n")
                f.write(f"- **Warnings**: {summary['warnings']}\n")
                f.write(f"- **Info Items**: {summary['info_issues']}\n")
                f.write(f"- **Total Issues**: {summary['total_issues']}\n\n")
                
                # Детальные проблемы
                if report['issues']:
                    f.write(f"## 📋 Detailed Issues\n\n")
                    
                    for issue in report['issues']:
                        severity_emoji = {
                            "critical": "🔴",
                            "warning": "🟡", 
                            "info": "ℹ️"
                        }
                        emoji = severity_emoji.get(issue['severity'], "❓")
                        
                        f.write(f"### {emoji} {issue['category'].title()} - {issue['severity'].title()}\n\n")
                        f.write(f"**Message**: {issue['message']}\n\n")
                        
                        if issue['value'] is not None:
                            f.write(f"**Current Value**: {issue['value']:.3f}\n\n")
                        
                        if issue['threshold'] is not None:
                            f.write(f"**Target/Threshold**: {issue['threshold']:.3f}\n\n")
                        
                        if issue['suggestion']:
                            f.write(f"**Suggestion**: {issue['suggestion']}\n\n")
                        
                        f.write("---\n\n")
                
                # Технические метрики
                f.write(f"## 📈 Technical Metrics\n\n")
                metrics = report['metrics']
                
                f.write(f"### Loudness Analysis\n")
                f.write(f"- **Integrated LUFS**: {metrics.get('lufs_integrated', 'N/A'):.1f}\n")
                f.write(f"- **Peak Level**: {metrics.get('peak_db', 'N/A'):.1f} dB\n")
                f.write(f"- **Dynamic Range**: {metrics.get('dynamic_range', 'N/A'):.1f} LU\n\n")
                
                if 'frequency_balance' in metrics:
                    balance = metrics['frequency_balance']
                    f.write(f"### Frequency Balance\n")
                    f.write(f"- **Low (0-300Hz)**: {balance['low']*100:.1f}%\n")
                    f.write(f"- **Mid (300Hz-3kHz)**: {balance['mid']*100:.1f}%\n") 
                    f.write(f"- **High (3kHz+)**: {balance['high']*100:.1f}%\n\n")
                
                if 'stereo_width' in metrics:
                    f.write(f"### Stereo Analysis\n")
                    f.write(f"- **Stereo Width**: {metrics['stereo_width']:.2f}\n")
                    f.write(f"- **Phase Correlation**: {metrics.get('phase_correlation', 'N/A'):.2f}\n\n")
                
                # Соответствие стандартам
                if 'compliance' in report:
                    f.write(f"## 📋 Standards Compliance\n\n")
                    
                    for standard, compliance in report['compliance'].items():
                        status = "✅ PASS" if compliance['overall'] else "❌ FAIL"
                        f.write(f"### {standard.upper()} {status}\n\n")
                        
                        details = compliance['details']
                        f.write(f"- **Loudness**: {details['lufs_actual']:.1f} LUFS (target: {details['lufs_target'][0]} to {details['lufs_target'][1]})\n")
                        f.write(f"- **Peak**: {details['peak_actual']:.1f} dB (max: {details['peak_max']})\n")
                        f.write(f"- **Dynamics**: {details['dr_actual']:.1f} LU (min: {details['dr_min']})\n\n")
                
                # Рекомендации
                f.write(f"## 💡 Recommendations\n\n")
                
                if summary['critical_issues'] > 0:
                    f.write("### Critical Actions Required\n")
                    f.write("- Review and fix all critical issues before release\n")
                    f.write("- Consider re-mastering if multiple critical issues exist\n\n")
                
                if summary['warnings'] > 0:
                    f.write("### Suggested Improvements\n")
                    f.write("- Address warning issues for optimal quality\n") 
                    f.write("- Test on various playback systems\n\n")
                
                f.write("### General Recommendations\n")
                f.write("- A/B test with reference tracks\n")
                f.write("- Check mono compatibility\n")
                f.write("- Test on different speakers/headphones\n")
                f.write("- Consider loudness normalization on target platforms\n\n")
                
                f.write("---\n")
                f.write("*Report generated by WaveDream Enhanced Pro Quality Analyzer*\n")
            
            self.logger.info(f"📋 Quality report saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error generating markdown report: {e}")
            return False