import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
INDEX_PATH = "D:/0/шаблоны/Samples for AKAI/sample_index.json"  # укажи путь к своему JSON
TAG_RULES = {
    "808": ["808"],
    "snare": ["snare", "snr", "snares"],
    "kick": ["kick", "bd", "bassdrum"],
    "hat": ["hat", "hihat", "hi-hat", "hh"],
    "clap": ["clap"],
    "perc": ["perc", "percussion"],
    "melody": ["mel", "lead", "synth", "guitar", "flute", "keys", "piano"],
    "pad": ["pad", "amb", "atmo"],
    "bass": ["bass", "sub"],
    "vocal": ["vox", "vocal", "phrase", "shout"],
    "fx": ["fx", "sweep", "rise", "impact", "reverse", "drop"]
}

TAG_PRIORITY = ["808", "kick", "snare", "hat", "clap", "perc", "bass", "melody", "pad", "vocal", "fx"]

def detect_tags(filename):
    found_tags = []
    lower = filename.lower()
    for tag in TAG_PRIORITY:
        keywords = TAG_RULES[tag]
        for kw in keywords:
            if kw in lower:
                found_tags.append(tag)
                break
    # Возвращаем только самый приоритетный тег из найденных, если хочешь 1 тег:
    if found_tags:
        return [found_tags[0]]
    else:
        return []

def build_index(sample_dir: str, output_path: str = INDEX_PATH):
    index = []
    for root, _, files in os.walk(sample_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                tags = detect_tags(file)
                entry = {
                    "path": path,
                    "tags": tags,
                    "filename": file
                }
                index.append(entry)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logging.info(f"✅ Индекс создан: {output_path} (файлов: {len(index)})")

if __name__ == "__main__":
    build_index("D:/0/шаблоны/Samples for AKAI", output_path=INDEX_PATH)

