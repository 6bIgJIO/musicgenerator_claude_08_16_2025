# smart_mixer
import os
import logging
from pydub import AudioSegment
from sample_picker import pick_samples
from SIG import load_index
from musicgen_wrapper import generate_music
import soundfile as sf
import uuid

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def smart_mix(json_data, sample_dir, output_dir="output", ignore_bpm=False):
    os.makedirs(output_dir, exist_ok=True)
    tempo = json_data.get("tempo", 120)
    structure = json_data.get("structure", [])
    tracks = json_data.get("tracks", [])

    total_duration_sec = sum([s["duration"] for s in structure])
    total_duration_ms = int(total_duration_sec * 1000)

    final_mix = AudioSegment.silent(duration=total_duration_ms)

    for track in tracks:
        name = track.get("name", "unknown")
        tags = track.get("sample_tags", [])
        volume = track.get("volume", -6)

        starts_at_beats = track.get("starts_at", 0)
        ends_at_beats = track.get("ends_at", None)

        beat_time = 60.0 / tempo
        starts_at_ms = int(starts_at_beats * beat_time * 1000)
        ends_at_ms = int(ends_at_beats * beat_time * 1000) if ends_at_beats else total_duration_ms

        duration_ms = ends_at_ms - starts_at_ms

        picked = pick_samples(tags, genre_hint=None, energy=0.5, tempo=tempo, sample_dir=sample_dir)

        if not picked:
            logging.warning(f"🧫 Fallback: генерим MusicGen stem для '{name}' (теги: {tags})")
            prompt = f"instrumental stem for {name} with tags {', '.join(tags)}"
            gen_duration = (ends_at_beats - starts_at_beats) if ends_at_beats else 8
            waveform, sr = generate_music(prompt, duration=gen_duration)
            mgen_path = os.path.join(output_dir, f"musicgen_{name}.wav")
            sf.write(mgen_path, waveform.squeeze().numpy(), sr)
            seg = AudioSegment.from_file(mgen_path)
        else:
            sample_path = picked[0]["path"]
            seg = AudioSegment.from_file(sample_path)

        # Подгонка длительности
        if len(seg) > duration_ms:
            seg = seg[:duration_ms]
        elif len(seg) < duration_ms:
            pad = AudioSegment.silent(duration=duration_ms - len(seg))
            seg += pad

        seg = seg.apply_gain(volume)
        final_mix = final_mix.overlay(seg, position=starts_at_ms)

        track_path = os.path.join(output_dir, f"{name}.wav")
        seg.export(track_path, format="wav")
        logging.info(f"🔊 [{name}] {starts_at_ms}ms → {ends_at_ms}ms  ({tags})")

    final_path = os.path.join(output_dir, "final_mix.wav")
    final_mix.export(final_path, format="wav")
    logging.info(f"🎛 Финал микс: {final_path}")
    return final_mix