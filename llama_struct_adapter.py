import os
import logging
from random import randint
from sample_aligner import align_sample_to_musicgen
from SIG import load_index

# Глобальные параметры
DEFAULT_TEMPO = 93
MEASURE_BEATS = 4

def parse_llama_json(json_data, base_wave_path, output_dir="output"):
    """
    Парсит JSON от LLaMA3 и вызывает align_sample_to_musicgen для генерации таймлайна.
    Гарантирует точное распределение треков по временной шкале, синхронно с длительностью.
    """
    tempo = json_data.get("tempo", DEFAULT_TEMPO)
    structure = json_data.get("structure", [])
    tracks = json_data.get("tracks", [])

    sample_index = load_index()
    timeline = []

    beat_time = 60.0 / tempo
    current_time = 0.0
    logging.info(f"[⚙️] Обработка структуры: {len(structure)} секций при {tempo} BPM")

    for section_idx, section in enumerate(structure):
        sec_type = section.get("type", f"section_{section_idx}")
        sec_duration_beats = section.get("duration", 16)
        sec_duration_secs = sec_duration_beats * beat_time

        logging.info(f"[📦] Секция '{sec_type}' ({sec_duration_beats} beats ≈ {sec_duration_secs:.1f} sec)")

        for track in tracks:
            track_name = track.get("name", "unknown")
            tags = track.get("sample_tags", [])
            volume = track.get("volume", -6)

            logging.info(f" └─ 🎚 Трек '{track_name}' | Теги: {tags} | Громкость: {volume}dB @ {current_time:.2f}s")

            result = align_sample_to_musicgen(
                base_path=base_wave_path,
                tags=tags,
                sample_index=sample_index,
                tempo=tempo,
                target_volume=volume
            )

            if result:
                segment, _ = result
                timeline.append({
                    "segment": segment,
                    "start_ms": int(current_time * 1000) + randint(0, 100),
                    "track": track_name,
                    "tags": tags,
                    "volume": volume,
                    "section": sec_type
                })

        current_time += sec_duration_secs

    return timeline