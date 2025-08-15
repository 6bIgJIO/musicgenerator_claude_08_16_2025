import os	
import random
from collections import defaultdict
from SIG import load_index


def fuzzy_match(tag, sample_tag):
    return tag in sample_tag or sample_tag in tag


def score_sample(sample, tags, energy, tempo, genre_hint=None):
    s_tags = sample.get("tags", [])
    s_tempo = sample.get("tempo", 120)

    tag_score = sum(3 for tag in tags for s_tag in s_tags if fuzzy_match(tag, s_tag))
    tempo_score = 2 if abs(s_tempo - tempo) <= 5 else 0
    genre_score = 2 if genre_hint and genre_hint.lower() in sample.get("path", "").lower() else 0

    return tag_score + tempo_score + genre_score + (energy * 2)


def pick_samples(tags, genre_hint=None, energy=0.5, tempo=120, sample_dir=None, top_n=5):
    """
    Подбор сэмплов по ключевым тегам и темпу.
    """
    index = load_index()
    matches = []

    # Перебор всех сэмплов
    for entry in index:
        entry_tags = entry.get("tags", [])
        entry_tempo = entry.get("tempo", 120)
        entry_category = entry.get("category", "oneshot")
        entry_path = entry.get("path")

        # Фильтруем по темпу и тегам
        tempo_ok = abs(entry_tempo - tempo) <= 6
        tag_match_score = sum([1.0 for tag in tags if tag in entry_tags])

        if tempo_ok and tag_match_score >= 1:
            matches.append((tag_match_score, entry))

    # Сортировка по релевантности
    matches.sort(key=lambda x: -x[0])

    if not matches:
        print(f"[⚠] Нет совпадений по тегам {tags} и темпу {tempo}")
        return []

    # Берём топ-N совпадений
    top_samples = [entry for _, entry in matches[:top_n]]
    print(f"[✓] Найдено сэмплов: {len(top_samples)} из {len(index)}")

    return top_samples


# Пример отладки (можно запускать напрямую)
if __name__ == "__main__":
    import json

    index = load_index()
    tags = ["trap_hat", "808", "snappy_snare", "trap_bell"]
    genre = "trap"

    result = pick_samples(tags, genre_hint=genre, energy=0.6, tempo=tempo, sample_dir=args.sample_dir)

    for tag, paths in result.items():
        print(f"\n🎯 {tag.upper()}:")
        for path in paths:
            print(f"  - {path}")