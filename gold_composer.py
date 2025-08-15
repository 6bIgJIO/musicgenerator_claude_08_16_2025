import os
import json
import logging
from pydub import AudioSegment
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from sample_picker import pick_samples
from SIG import load_index
from smart_mixer import smart_mix
from self_check import verify_mix
from mistral_client import query_structured_music

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def generate_full_track(prompt, sample_dir, output_dir, model_name="facebook/musicgen-medium", professional=True):
    report_path = os.path.join(output_dir, "report.json")
    stems_dir = os.path.join(output_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    # === Шаг 1. Загрузка JSON структуры от Llama3 ===
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    tempo = report["tempo"]
    structure = report["structure"]
    tracks = report["tracks"]

    # === Шаг 2. Инициализация MusicGen ===
    logging.info(f"🎧 Загрузка модели MusicGen: {model_name}")
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(use_sampling=True, top_k=250, temperature=1.2)

    # === Шаг 3. Генерация MusicGen по секциям ===
    final_mix = AudioSegment.silent(duration=0)
    for i, part in enumerate(structure):
        duration = part["duration"]
        section_type = part["type"]
        tags = sorted(set(tag for track in tracks for tag in track.get("sample_tags", [])))
        prompt_section = f"{section_type} section, tempo {tempo}, style: {' '.join(tags)}"

        logging.info(f"🎼 Генерация {section_type.upper()} ({duration}s) → {prompt_section}")
        model.set_generation_params(duration=duration)
        wav = model.generate([prompt_section])

        part_path = os.path.join(stems_dir, f"{i:02d}_{section_type}.wav")
        audio_write(part_path.replace(".wav", ""), wav[0].cpu(), model.sample_rate, strategy="loudness")
        final_mix += AudioSegment.from_wav(part_path)

    musicgen_path = os.path.join(stems_dir, "musicgen_original.wav")
    final_mix.export(musicgen_path, format="wav")
    logging.info(f"✅ Базовый MusicGen трек сохранён: {musicgen_path}")

    # === Шаг 4. Подбор сэмплов из sample_dir (кастомизация) ===
    logging.info("🔍 Подбор кастомных сэмплов по тегам...")
    try:
        index = load_index()
    except:
        logging.error("❌ Индекс сэмплов не загружен")
        return

    picked_tracks = []
    for track in tracks:
        picked = pick_samples(
            index,
            sample_tags=track["sample_tags"],
            genre_hint=None,
            top_k=1,
            energy=0.6,
            tempo=tempo
        )
        if picked:
            track["path"] = picked[0]
            picked_tracks.append(track)
        else:
            logging.warning(f"⚠️ Не найден сэмпл для: {track['name']}")

    # === Шаг 5. Смарт-микс и экспорт пользовательского микса ===
    logging.info("🎛 Смарт-миксинг кастомных сэмплов...")
    smart_mix(report, sample_dir, output_dir=stems_dir)

    custom_mix_path = os.path.join(stems_dir, "custom_mix.wav")
    if not os.path.exists(custom_mix_path):
        logging.error("❌ Не удалось сгенерировать custom_mix.wav")
        return

    # === Шаг 6. Проверка качества и соответствия кастомного трека ===
    if professional:
        logging.info("🤖 Проверка соответствия кастомного трека промпту...")
        result = verify_mix(custom_mix_path, prompt)
        if not result["ok"]:
            logging.warning(f"⚠️ Проверка не пройдена: {result['reason']}")
            # Попытка исправить с помощью Llama3 (или другой модели)
            logging.info("🛠 Запрос на правку трека...")
            fixed_path = os.path.join(stems_dir, "custom_fixed.wav")
            response = query_structured_music(prompt, audio_path=custom_mix_path, fix_mode=True)
            if response and os.path.exists(response):
                os.rename(response, fixed_path)
                logging.info(f"✅ Исправленный трек сохранён: {fixed_path}")
            else:
                logging.error("❌ Не удалось получить исправление от Llama3")
        else:
            logging.info("✅ Проверка пройдена успешно, трек готов к использованию.")

    logging.info("🎉 Все этапы завершены.")