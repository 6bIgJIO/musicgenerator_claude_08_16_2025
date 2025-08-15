import os
from pydub import AudioSegment
import logging

def overlay_timeline(base_wav_path, timeline, output_path="out/final_mix.wav"):
    """
    Собирает финальный микс из базового трека и списка вставок (timeline).
    """
    if not os.path.exists(base_wav_path):
        raise FileNotFoundError(f"❌ Основной WAV не найден: {base_wav_path}")

    base = AudioSegment.from_file(base_wav_path).set_channels(2).set_sample_width(2)
    logging.info(f"[🎼] Загружена основа: {len(base)/1000:.1f} сек")

    for i, item in enumerate(timeline):
        seg = item["segment"]
        pos = item["start_ms"]

        try:
            logging.info(f"  └─ [+] Вставка #{i+1} @ {pos} мс | трек: {item['track']} | теги: {item['tags']}")
            base = base.overlay(seg, position=pos)
        except Exception as e:
            logging.warning(f"⚠️ Не удалось вставить сегмент #{i+1} → {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base.export(output_path, format="wav")
    logging.info(f"[✅] Финальный микс сохранён: {output_path}")
    return output_path