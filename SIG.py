import os
import json
import librosa
import argparse
import subprocess
from pydub import AudioSegment

SAMPLE_DIR = r"D:\0\шаблоны\Samples for AKAI"
INDEX_FILE = os.path.join(SAMPLE_DIR, "sample_index.json")

def is_edison_ogg_wav(file_path):
    """Проверка: битый WAV от Edison с OGG внутри"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", file_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
        )
        return "Og[0][0]" in result.stderr
    except Exception as e:
        print(f"[!] Не удалось проверить {file_path}: {e}")
        return False

def fix_wav_file(file_path):
    """Перекодирует OGG-in-WAV (Edison/Slicex) в нормальный WAV"""
    base, _ = os.path.splitext(file_path)
    backup = base + ".bak"
    fixed = base + "_fixed.wav"

    try:
        os.rename(file_path, backup)
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "ogg", "-i", backup,
            "-c:a", "pcm_s16le", "-f", "wav",
            fixed
        ], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

        if os.path.exists(fixed):
            os.replace(fixed, file_path)
            print(f"[✅] Исправлен: {file_path}")
            return True
        else:
            os.rename(backup, file_path)
            print(f"[❌] Не удалось исправить: {file_path}")
            print(result.stderr)  # отладка
            return False
    except Exception as e:
        print(f"[!!] Ошибка при исправлении {file_path}: {e}")
        return False

def fix_all_edison_samples():
    print("[🛠] Поиск и исправление битых WAV...")
    count_fixed = 0
    for root, _, files in os.walk(SAMPLE_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                if is_edison_ogg_wav(full_path):
                    print(f"[💀] Найден битый: {full_path}")
                    if fix_wav_file(full_path):
                        count_fixed += 1
    print(f"[✓] Исправлено файлов: {count_fixed}")

def extract_metadata(sample_path):
    try:
        audio = AudioSegment.from_file(sample_path)
        duration = len(audio) / 1000
        return {
            "path": sample_path,
            "tempo": 120,
            "duration": float(duration),
            "key": None
        }
    except Exception as e:
        print(f"[!] Ошибка при чтении {sample_path}: {e}")
        return None

def index_samples():
    fix_all_edison_samples()  # 💉 встроенный фикс перед индексацией

    all_samples = []
    for root, _, files in os.walk(SAMPLE_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                print(f"[+] Анализ {file}")
                metadata = extract_metadata(full_path)
                if metadata:
                    all_samples.append(metadata)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"[✓] Сохранён индекс в {INDEX_FILE}")

def load_index():
    if not os.path.exists(INDEX_FILE):
        print("[!] Индекс не найден, запускаем индексацию...")
        index_samples()
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def find_matching_samples(query_bpm, tolerance=5):
    samples = load_index()
    matches = [s for s in samples if abs(s["tempo"] - query_bpm) <= tolerance]
    print(f"[✓] Найдено совпадений с BPM ~{query_bpm}: {len(matches)}")
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIG.py — индексатор и фильтратор WAV-сэмплов по BPM")
    parser.add_argument("--index", action="store_true", help="Проиндексировать сэмплы заново")
    parser.add_argument("--find", type=float, help="Найти сэмплы по BPM")
    args = parser.parse_args()

    if args.index:
        index_samples()
    elif args.find:
        matches = find_matching_samples(args.find)
        for sample in matches:
            print(f'🎵 {sample["path"]} | {sample["tempo"]} BPM | {sample["duration"]:.1f} sec')
    else:
        parser.print_help()