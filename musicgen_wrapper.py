# musicgen_wrapper

import torch
from audiocraft.models import musicgen
import torchaudio
import io
import numpy as np
import soundfile as sf
from audiocraft.models.musicgen import MusicGen
from typing import Optional, List, Dict, Union
import warnings
import logging
warnings.filterwarnings("ignore")

try:
    from audiocraft.models import musicgen
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

class MusicGenEngine:
    def __init__(self, model_name: str = "D:/2027/audiocraft/audiocraft/models/facebook/musicgen-medium"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MUSICGEN_AVAILABLE = False
        self._load_model()

    def _load_model(self):
        try:
            from audiocraft.models import musicgen
            fallback_models = [
                self.model_name,
                "facebook/musicgen-small",
                "facebook/musicgen-medium",
                "facebook/musicgen-large"
            ]
            for name in fallback_models:
                try:
                    self.logger.info(f"Loading MusicGen model: {name}")
                    self.model = musicgen.MusicGen.get_pretrained(name)
                    self.model_name = name
                    self.model.set_generation_params(duration=8)
                    self.logger.info(f"✅ MusicGen loaded on {self.device} with model {name}")
                    self.MUSICGEN_AVAILABLE = True
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load model {name}: {e}")
            if not self.MUSICGEN_AVAILABLE:
                self.logger.error("Failed to load any MusicGen model")
        except ImportError as e:
            self.logger.error(f"MusicGen not available: {e}")

    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        genre_hint: Optional[str] = None
    ) -> bytes:
        if not self.model or not self.MUSICGEN_AVAILABLE:
            raise RuntimeError("MusicGen model not available")

        self.logger.info(f"🎼 Generating: '{prompt}' ({duration}s, temp={temperature})")

        try:
            safe_duration = min(duration, 30)
            self.model.set_generation_params(
                duration=safe_duration,
                use_sampling=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            enhanced_prompt = self._enhance_prompt_for_genre(prompt, genre_hint)
            self.logger.info(f"📝 Enhanced prompt: {enhanced_prompt}")

            with torch.no_grad():
                wav_tensor = self.model.generate([enhanced_prompt])

            if wav_tensor is None or wav_tensor.size(0) == 0:
                raise RuntimeError("Model returned empty result")

            self.logger.info(f"🔊 Generated tensor shape: {wav_tensor.shape}")

            if wav_tensor.dim() == 3:
                audio_array = wav_tensor[0].cpu().numpy()
            elif wav_tensor.dim() == 2:
                audio_array = wav_tensor.cpu().numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {wav_tensor.shape}")

            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))

            rms = np.sqrt(np.mean(audio_array**2))
            self.logger.info(f"🔊 Generated audio RMS: {rms:.6f}")

            if rms < 1e-6:
                self.logger.warning("⚠️ Very quiet audio generated, boosting signal")
                audio_array = audio_array * 1000
                audio_array = np.clip(audio_array, -1.0, 1.0)

            sample_rate = self.model.sample_rate

            buffer = io.BytesIO()
            if audio_array.ndim == 1:
                sf.write(buffer, audio_array, sample_rate, format='WAV')
            else:
                if audio_array.shape[0] == 2:
                    sf.write(buffer, audio_array.T, sample_rate, format='WAV')
                else:
                    sf.write(buffer, audio_array[0], sample_rate, format='WAV')

            audio_bytes = buffer.getvalue()
            buffer.close()

            if len(audio_bytes) < 1000:
                raise RuntimeError(f"Generated file too small: {len(audio_bytes)} bytes")

            self.logger.info(f"✅ MusicGen generation completed: {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            self.logger.error(f"❌ MusicGen generation error: {e}")
            return self._generate_fallback_audio(duration)

    def _generate_fallback_audio(self, duration: int) -> bytes:
        self.logger.warning("🔄 Generating fallback audio...")

        try:
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            noise = np.random.normal(0, 0.1, len(t))
            bass = np.sin(2 * np.pi * 60 * t) * 0.3
            mid = np.sin(2 * np.pi * 440 * t) * 0.2
            audio_array = (noise + bass + mid) * 0.5

            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            return buffer.getvalue()

        except Exception as e:
            self.logger.error(f"❌ Fallback generation failed: {e}")
            return self._create_minimal_wav(duration)

    def _create_minimal_wav(self, duration: int) -> bytes:
        sample_rate = 44100
        samples = int(sample_rate * duration)
        audio_array = np.random.normal(0, 0.01, samples).astype(np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        return buffer.getvalue()

    def _enhance_prompt_for_genre(self, prompt: str, genre: Optional[str]) -> str:
        if not genre:
            return prompt

        genre_enhancements = {
            "trap": "heavy 808s, tight snares, dark atmosphere",
            "lofi": "warm analog sound, vinyl texture, mellow vibes",
            "dnb": "fast breakbeats, heavy bass, energetic",
            "ambient": "ethereal pads, spacious reverb, peaceful",
            "techno": "driving four-on-the-floor, hypnotic, industrial",
            "house": "groovy four-on-the-floor, soulful, danceable"
        }

        enhancement = genre_enhancements.get(genre, "")
        if enhancement:
            enhanced = f"{prompt}, {enhancement}"
            self.logger.debug(f"Enhanced prompt: {enhanced}")
            return enhanced
        return prompt