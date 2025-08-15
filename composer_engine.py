import os
import random
import re
import time
import json
import argparse
import logging
import uuid
import subprocess
import requests
import shutil
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import mediainfo
from smart_mixer import smart_mix
from fl_builder import build_flp_project
from fl_launcher import launch_fl, focus_fl, import_samples, drag_wav_to_playlist
from fl_midi_gen import generate_midi_from_structured_json
from fl_midi import import_midi_to_fl
from mistral_client import query_structured_music
from SIG import index_samples, load_index
from wav_renderer import render_wav
from sample_picker import pick_samples
from sample_aligner import align_sample_to_musicgen
from self_check import verify_mix
from musicgen_wrapper import generate_music
from composer.instrument_tag_mapper import instrument_to_tags, expand_tags
from composer.structure import generate_structure
from composer.prompt_parser import extract_additional_tags
from composer.genre_detector import detect_genre
from composer.instrument_selector import select_instruments
from composer.output_renderer import export_track
import traceback


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def is_llama3_music_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        tags = response.json()
        return any(tag["name"] == "llama3-music:latest" for tag in tags.get("models", []))
    except Exception as e:
        logging.warning(f"⚠️ Ollama недоступен: {e}")
        return False

def ensure_llama3_music_running():
    if not shutil.which("ollama"):
        logging.error("❌ Не найден исполняемый файл ollama. Проверь установку.")
        return False

    if is_llama3_music_running():
        logging.info("✅ Модель llama3:music уже работает.")
        return True

    logging.info("⏳ Запускаю модель llama3:music через Ollama...")
    try:
        subprocess.Popen(["ollama", "run", "llama3-music:latest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        for _ in range(10):
            if is_llama3_music_running():
                logging.info("✅ Модель успешно запущена.")
                return True
            time.sleep(1)

        logging.error("❌ Не удалось запустить модель llama3:music.")
        return False
    except Exception as e:
        logging.error(f"❌ Ошибка запуска модели: {e}")
        return False

def generate_structure_from_prompt(prompt):
    return query_structured_music("llama3-music:latest", prompt)

def main(prompt, sample_dir, output_dir, mode):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"📡 Старт генерации ({mode})")

    if mode != "hybrid":
        logging.error("❌ Только режим hybrid сейчас поддерживается.")
        return

    if not ensure_llama3_music_running():
        return

    # Запрос к LLaMA3
    model = "llama3-music:latest"
    json_data = query_structured_music(model, prompt)
    if not json_data:
        logging.error("❌ Ошибка получения JSON от LLaMA3")
        return

    # Сохраняем JSON
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logging.info(f"🧾 JSON сохранён: {json_path}")

    # Расчёт длительности
    duration_sec = sum([s["duration"] for s in json_data.get("structure", [])])
    duration_ms = int(duration_sec * 1000)

    # Генерация MusicGen WAV
    waveform, sr = generate_music(prompt, duration=int(duration_sec))
    path_to_musicgen_output = os.path.join(output_dir, "musicgen.wav")
    sf.write(path_to_musicgen_output, waveform.squeeze().numpy(), sr, format="WAV")
    logging.info(f"🎵 MusicGen завершён: {path_to_musicgen_output}")

    # Сборка stem микса с таймлайном
    stem_dir = os.path.join(output_dir, "stems")
    final_mix = smart_mix(json_data, sample_dir, output_dir=stem_dir)

    # Подмешиваем MusicGen в микс (без искажения длины)
    musicgen_seg = AudioSegment.from_file(path_to_musicgen_output) - 6
    if len(final_mix) < duration_ms:
        final_mix += AudioSegment.silent(duration=duration_ms - len(final_mix))
    final_mix = final_mix.overlay(musicgen_seg)

    final_path = os.path.join(output_dir, "final_mix.wav")
    final_mix.export(final_path, format="wav")
    logging.info(f"💾 Финальный микс сохранён: {final_path}")

    result = verify_mix(final_path)
    if not result["ok"]:
        logging.warning(f"⚠️ Проблема в финальном миксе: {result['reason']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎼 Composer Engine")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--mode", type=str, choices=["hybrid"], default="hybrid")
    args = parser.parse_args()
    main(args.prompt, args.sample_dir, args.output_dir, args.mode)