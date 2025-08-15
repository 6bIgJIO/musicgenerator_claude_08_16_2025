import os
import json
import subprocess
import re
import warnings
from pydub import AudioSegment
import librosa
from composer.instrument_tag_mapper import instrument_to_tags

warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_DIR = r"D:\0\шаблоны\Samples for AKAI"
INDEX_FILE = os.path.join(SAMPLE_DIR, "sample_index.json")

try:
    from composer.instrument_tag_mapper import instrument_to_tags
    TAG_RULES = {tag: [kw.lower() for kw in keywords] for tag, keywords in instrument_to_tags.items()}
except Exception as e:
    print(f"[⚠] instrument_to_tags не найден, используем дефолтный набор: {e}")
    TAG_RULES = {
        "808": ["808", "sub"],
        "kick": ["kick", "bd", "bass drum"],
        "snare": ["snare", "snr"],
        "hat": ["hi hat", "hat", "hh"],
        "clap": ["clap"],
        "perc": ["perc", "percussion"],
        "bass": ["bass", "sub"],
        "melody": ["mel", "lead", "synth", "guitar", "keys", "flute", "piano"],
        "pad": ["pad", "atmo", "ambient"],
        "vocal": ["vox", "vocal", "phrase", "shout"],
        "fx": ["fx", "sweep", "impact", "reverse", "drop"]
    }

def score_tags_from_filename(filename):
    lower = filename.lower()
    tag_scores = {}
    category = "oneshot"

    for tag, keywords in TAG_RULES.items():
        for kw in keywords:
            if kw in lower:
                tag_scores[tag] = tag_scores.get(tag, 0.0) + 1.0

    if re.search(r"\b(loop|stem|phrase|break|full)\b", lower):
        category = "loop"

    return tag_scores, category

def is_edison_ogg_wav(path):
    try:
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", path],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                text=True)
        return "Og[0][0]" in result.stderr
    except:
        return False


def fix_wav(path):
    base, _ = os.path.splitext(path)
    backup = base + ".bak"
    fixed = base + "_fixed.wav"
    try:
        os.rename(path, backup)
        subprocess.run(["ffmpeg", "-y", "-f", "ogg", "-i", backup,
                        "-c:a", "pcm_s16le", "-f", "wav", fixed],
                       stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        if os.path.exists(fixed):
            os.replace(fixed, path)
            print(f"[FIXED] {path}")
            return True
        else:
            os.rename(backup, path)
            print(f"[FAIL] {path}")
            return False
    except Exception as e:
        print(f"[ERR] {path} → {e}")
        return False


def analyze_key(y, sr):
    try:
        if len(y) < 1024:
            return None
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = chroma_mean.argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return keys[key_index]
    except Exception as e:
        print(f"[WARN] Анализ тональности не удался: {e}")
        return None


def build_index(sample_dir=SAMPLE_DIR, output_path=INDEX_FILE):
    index = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                index = {entry["path"]: entry for entry in json.load(f)}
            except:
                print("[!] Старый индекс битый или пустой")

    for root, _, files in os.walk(sample_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                print(f"[🔍] Обработка: {file}")

                if is_edison_ogg_wav(path):
                    fix_wav(path)

                try:
                    print("  └─ [🧪] AudioSegment.from_file...")
                    audio = AudioSegment.from_file(path)
                    duration = len(audio) / 1000
                except Exception as e:
                    print(f"[SKIP] {path} (AudioSegment fail): {e}")
                    continue

                try:
                    print("  └─ [🎧] librosa.load...")
                    y, sr = librosa.load(path, sr=None, mono=True, duration=10.0)
                except Exception as e:
                    print(f"[SKIP] {path} (librosa.load fail): {e}")
                    continue

                tag_scores, category = score_tags_from_filename(file)
                top_tags = sorted(tag_scores.items(), key=lambda x: -x[1])

                entry = index.get(path, {})
                entry.update({
                    "path": path,
                    "filename": file,
                    "tempo": 120,
                    "duration": round(duration, 3),
                    "key": analyze_key(y, sr),
                    "category": category,
                    "tag_scores": tag_scores,
                    "tags": [tag for tag, _ in top_tags[:2]]
                })

                index[path] = entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(index.values()), f, indent=2, ensure_ascii=False)
        print(f"[✓] Индекс обновлён: {output_path} (всего: {len(index)})")


if __name__ == "__main__":
    build_index()