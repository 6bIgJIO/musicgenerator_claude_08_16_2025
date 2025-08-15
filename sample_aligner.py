import os
import uuid
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import pyrubberband as pyrb
import logging
from SIG import load_index
from sample_picker import pick_samples


def align_sample_to_musicgen(base_path, tags, sample_index=None, tempo=120, target_volume=-6, sample_dir=None):
    try:
        y_base, sr = librosa.load(base_path, sr=None)
        tuning_base = librosa.estimate_tuning(y=y_base, sr=sr)
    except Exception as e:
        logging.warning(f"⚠️ Ошибка анализа основы: {e}")
        return None

    picked = pick_samples(
        tags=tags,
        genre_hint=None,
        energy=0.7,
        tempo=int(tempo),
        sample_dir=sample_dir
    )

    if not picked:
        logging.warning(f"🪫 Нет сэмплов для тегов: {tags}")
        return None

    sample_path = picked[0]["path"]
    try:
        y_sample, sr_sample = librosa.load(sample_path, sr=sr)
        tuning_sample = librosa.estimate_tuning(y=y_sample, sr=sr)
        semitone_shift = round((tuning_base - tuning_sample) * 12)

        if semitone_shift != 0:
            y_shifted = pyrb.pitch_shift(y_sample, sr, n_steps=semitone_shift)
        else:
            y_shifted = y_sample

        temp_wav = f"temp_shifted_{uuid.uuid4().hex[:6]}.wav"
        sf.write(temp_wav, y_shifted, sr)

        seg = AudioSegment.from_file(temp_wav) + target_volume

        logging.info(f"🎯 Сэмпл '{os.path.basename(sample_path)}' сдвинут на {semitone_shift} полутонов")
        return seg.set_frame_rate(sr).set_channels(2).set_sample_width(2).fade_in(10).fade_out(20), 0

    except Exception as e:
        logging.warning(f"⚠️ Ошибка обработки сэмпла {sample_path}: {e}")
        return None