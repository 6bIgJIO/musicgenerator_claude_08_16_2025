# mistral_client
import requests
import json
import subprocess
import logging

def query_structured_music(prompt: str) -> dict:
    """
    ИСПРАВЛЕННЫЙ запрос к LLM с вашими моделями
    """
    
    # Сначала пробуем через HTTP API (быстрее и надежнее)
    result = _query_ollama_api(prompt)
    if result:
        return result
    
    # Fallback на subprocess с оптимизацией
    return _query_ollama_subprocess(prompt)


def _query_ollama_api(prompt: str) -> dict:
    """
    Запрос через Ollama HTTP API (рекомендуемый способ)
    """
    llama_prompt = """
Ты — генератор структуры музыкального трека в формате JSON.

Сгенерируй ТОЛЬКО JSON по шаблону:

{
  "BPM": !В завимисимости от жанра выбранного user'ом,
  "structure": [
    {"type": "intro", "duration": 8++},
    {"type": "buildup", "duration": 16++},
    {"type": "builddown", "duration": 16++},
    {"type": "outro", "duration": 8++}
  ],
  "tracks": [
    {
      "name": "kick",
      "sample_tags": ["kick", "bass", "drum"],
      "volume": -3
    }
    {
      "name": "user's choice",
      "sample_tags": ["lead", "synth", "drum"],
      "volume": -4
    }
и т.д...
  ]
}

❗ Только JSON. Никаких пояснений.
❗ Используй двойные кавычки.
❗ НЕ добавляй комментарии в JSON.
""".strip()

    full_prompt = f"{llama_prompt}\n\nОписание трека: {prompt.strip()}"
    
    # ВАШИ РЕАЛЬНЫЕ МОДЕЛИ в порядке приоритета
    models_to_try = [
        "llama3-music:latest",    # Ваша специализированная модель
        "llama3:latest",          # Основная модель
        "llama3.2:latest",        # Более новая версия
        "mistral:7b",             # Альтернатива
        "mistral:latest"          # Fallback
    ]
    
    for model in models_to_try:
        try:
            logging.info(f"🚀 Пробуем модель {model} через Ollama API")
            
            response = requests.post(
                "http://localhost:11434/api/tags",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Очень низкая для JSON
                        "top_p": 0.5,
                        "num_predict": 1000,   # Ограничиваем количество токенов
                        "stop": ["```", "---", "Explanation:", "Note:", "\n\n\n"]
                    }
                },
                timeout=250  # Увеличиваем таймаут для первой модели
            )
            
            if response.status_code == 200:
                result_data = response.json()
                raw_output = result_data.get("response", "")
                
                if raw_output.strip():
                    logging.info(f"✅ Получен ответ от {model}")
                    return _parse_json_from_response(raw_output)
                    
        except requests.exceptions.ConnectionError:
            logging.warning(f"⚠️ Ollama API недоступна")
            break  # Не пробуем другие модели если API недоступна
        except requests.exceptions.Timeout:
            logging.warning(f"⚠️ Таймаут для модели {model}")
            continue
        except Exception as e:
            logging.warning(f"⚠️ Ошибка с моделью {model}: {e}")
            continue
    
    return None


def _query_ollama_subprocess(prompt: str) -> dict:
    """
    Fallback через subprocess (ваш оригинальный код, но исправленный)
    """
    llama_prompt = """
Ты — генератор структуры музыкального трека в формате JSON.

Сгенерируй ТОЛЬКО JSON по шаблону:

{
  "BPM": !В завимисимости от жанра выбранного user'ом,
  "structure": [
    {"type": "intro", "duration": 8++},
    {"type": "buildup", "duration": 16++},
    {"type": "builddown", "duration": 16++},
    {"type": "outro", "duration": 8++}
  ],
  "tracks": [
    {
      "name": "kick",
      "sample_tags": ["kick", "bass", "drum"],
      "volume": -3
    }
    {
      "name": "user's choice",
      "sample_tags": ["lead", "synth", "drum"],
      "volume": -4
    }
...
  ]
}
или под структуру пользователя, но ни в коем случае ни комментариями и никакими подобными пояснениями, ты молчаливый композитор!
❗ Только JSON. Никаких пояснений.
❗ Используй двойные кавычки.
❗ НЕ добавляй комментарии в JSON.
""".strip()

    full_prompt = f"{llama_prompt}\n\nОписание трека: {prompt.strip()}"
    
    models_to_try = [
        "llama3-music:latest",    # Ваша специализированная модель
        "llama3:latest",          # Основная 
        "llama3.2:latest",        # Более компактная
        "mistral:7b",             # Быстрая альтернатива
        "mistral:latest"          # Последний fallback
    ]
    
    for model in models_to_try:
        try:
            logging.info(f"🚀 Запрос структуры к {model} через ollama run")

            result = subprocess.run(
                ['ollama', 'run', model],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120 if "llama3-music" in model else 60  # Больше времени для специализированной модели
            )

            if result.returncode != 0:
                logging.warning(f"⚠️ Ошибка ollama для {model}: {result.stderr}")
                continue

            raw_output = result.stdout.strip()
            
            if raw_output:
                logging.info(f"✅ Получен ответ от {model}")
                return _parse_json_from_response(raw_output)

        except subprocess.TimeoutExpired:
            logging.warning(f"⚠️ Запрос к {model} превысил лимит времени")
            continue
        except Exception as e:
            logging.warning(f"⚠️ Неожиданная ошибка с {model}: {e}")
            continue

    logging.error("❌ Все попытки запроса к LLM провалились")
    return None


def _parse_json_from_response(raw_output: str) -> dict:
    """
    УЛУЧШЕННЫЙ парсинг JSON из ответа LLM
    """
    logging.debug(f"Парсим ответ: {raw_output[:200]}...")
    
    # Список кандидатов JSON
    json_candidates = []
    
    # Метод 1: Поиск по маркерам ```json
    if "```json" in raw_output.lower():
        start = raw_output.lower().find("```json") + 7
        end = raw_output.find("```", start)
        if end != -1:
            json_candidates.append(raw_output[start:end].strip())
    
    # Метод 2: Поиск первого { до последнего }
    first_brace = raw_output.find('{')
    last_brace = raw_output.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        json_candidates.append(raw_output[first_brace:last_brace+1])
    
    # Метод 3: Построчный анализ с подсчетом скобок
    lines = raw_output.split('\n')
    current_json = ""
    brace_count = 0
    in_json = False
    
    for line in lines:
        stripped_line = line.strip()
        
        # Пропускаем комментарии
        if stripped_line.startswith(('//:', '#', '/*', '*/', 'Note:', 'Explanation:')):
            continue
            
        if '{' in stripped_line and not in_json:
            in_json = True
            current_json = ""
        
        if in_json:
            current_json += line + '\n'
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and '}' in line:
                json_candidates.append(current_json.strip())
                in_json = False
                current_json = ""
                break
    
    # Пробуем парсить кандидатов
    for i, candidate in enumerate(json_candidates):
        try:
            # Очищаем от лишнего
            cleaned = _clean_json_text(candidate)
            if not cleaned:
                continue
                
            parsed = json.loads(cleaned)
            
            # Валидируем структуру
            if _validate_llama_response(parsed):
                logging.info(f"✅ JSON успешно распарсен (метод {i+1})")
                return _normalize_llama_response(parsed)
        
        except json.JSONDecodeError as e:
            logging.debug(f"⚠️ JSON кандидат {i+1} не валиден: {e}")
            continue
        except Exception as e:
            logging.debug(f"⚠️ Ошибка обработки кандидата {i+1}: {e}")
            continue
    
    logging.error("❌ Не удалось найти валидный JSON в ответе LLM")
    return None


def _clean_json_text(text: str) -> str:
    """
    Очистка JSON текста от мусора
    """
    if not text:
        return ""
    
    # Убираем markdown разметку
    text = text.replace('```json', '').replace('```', '')
    
    # Убираем комментарии в стиле //
    lines = []
    for line in text.split('\n'):
        # Проверяем, не является ли строка комментарием
        stripped = line.strip()
        if not stripped.startswith(('//:', '//', '#')):
            lines.append(line)
    
    text = '\n'.join(lines)
    
    # Убираем trailing commas (частая проблема LLM)
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text.strip()


def _validate_llama_response(data: dict) -> bool:
    """
    Валидация структуры ответа от LLaMA
    """
    if not isinstance(data, dict):
        return False
    
    # Проверяем обязательные поля
    if 'tempo' not in data:
        return False
    
    if 'structure' not in data or not isinstance(data['structure'], list):
        return False
    
    # Проверяем структуру секций
    for section in data['structure']:
        if not isinstance(section, dict):
            return False
        if 'type' not in section or 'duration' not in section:
            return False
    
    return True


def _normalize_llama_response(data: dict) -> dict:
    """
    Нормализация ответа от LLaMA к стандартному формату
    """
    normalized = {
        'tempo': int(data.get('bpm', 120)),
        'structure': [],
        'tracks': data.get('tracks', [])
    }
    
    # Нормализуем структуру
    for section in data.get('structure', []):
        normalized_section = {
            'type': section.get('type', 'section').lower(),
            'duration': int(section.get('duration', 16))
        }
        normalized['structure'].append(normalized_section)
    
    # Если структура пустая, создаем базовую
    if not normalized['structure']:
        normalized['structure'] = [
            {'type': 'intro', 'duration': 8},
            {'type': 'verse', 'duration': 16},
            {'type': 'hook', 'duration': 16},
            {'type': 'outro', 'duration': 8}
        ]
    