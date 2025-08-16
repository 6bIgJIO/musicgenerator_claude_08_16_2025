# mistral_client.py - ИСПРАВЛЕННАЯ VERSION
import requests
import json
import subprocess
import logging
import re

def query_structured_music(prompt: str) -> dict:
    """
    ИСПРАВЛЕННЫЙ запрос к LLM для генерации структуры музыки
    """
    
    # Сначала пробуем через HTTP API (быстрее и надежнее)
    result = _query_ollama_api(prompt)
    if result:
        logging.info(f"✅ Структура получена через API: {len(result.get('structure', []))} секций")
        return result
    
    # Fallback на subprocess с оптимизацией
    result = _query_ollama_subprocess(prompt)
    if result:
        logging.info(f"✅ Структура получена через subprocess: {len(result.get('structure', []))} секций")
        return result
    
    logging.error("❌ Все попытки получения структуры провалились")
    return None


def _query_ollama_api(prompt: str) -> dict:
    """
    ИСПРАВЛЕННЫЙ запрос через Ollama HTTP API
    """
    # ИСПРАВЛЕННЫЙ промпт без комментариев в JSON
    llama_prompt = f"""Generate a music track structure in JSON format for: {prompt}

Return only valid JSON with this exact structure:

{{
  "BPM": 120,
  "structure": [
    {{"type": "intro", "duration": 8}},
    {{"type": "buildup", "duration": 16}},
    {{"type": "drop", "duration": 16}},
    {{"type": "outro", "duration": 8}}
  ]
}}

Rules:
- Only JSON, no explanations
- Use double quotes only
- Match the genre and mood from description
- Total duration should be 60-120 seconds
- Section types: intro, verse, chorus, buildup, drop, breakdown, bridge, outro"""

    # ИСПРАВЛЕН: Проверяем доступность Ollama API
    try:
        ping_response = requests.get("http://localhost:11434/api/version", timeout=5)
        if ping_response.status_code != 200:
            logging.warning("⚠️ Ollama API недоступна")
            return None
    except:
        logging.warning("⚠️ Ollama API недоступна")
        return None
    
    # ВАШИ РЕАЛЬНЫЕ МОДЕЛИ в порядке приоритета
    models_to_try = [
        "llama3-music:latest",    # Ваша специализированная модель
        "llama3.1:latest",        # Стабильная версия
        "llama3:latest",          # Основная модель
        "llama3.2:latest",        # Более новая версия
        "mistral:7b",             # Альтернатива
        "mistral:latest"          # Fallback
    ]
    
    for model in models_to_try:
        try:
            logging.info(f"🚀 Пробуем модель {model} через Ollama API")
            
            # ИСПРАВЛЕН: Правильный endpoint /api/tags
            response = requests.post(
                "http://localhost:11434/api/generate",  # ИСПРАВЛЕНО!
                json={
                    "model": model,
                    "prompt": llama_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Очень низкая для JSON
                        "top_p": 0.5,
                        "num_predict": 500,   # Ограничиваем количество токенов
                        "stop": ["```", "---", "Explanation:", "Note:", "\n\n\n", "Human:", "Assistant:"]
                    }
                },
                timeout=90  # Разумный таймаут
            )
            
            if response.status_code == 200:
                result_data = response.json()
                raw_output = result_data.get("response", "")
                
                if raw_output.strip():
                    logging.info(f"✅ Получен ответ от {model}: {len(raw_output)} символов")
                    parsed_result = _parse_json_from_response(raw_output)
                    if parsed_result:
                        return parsed_result
                        
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
    ИСПРАВЛЕННЫЙ fallback через subprocess
    """
    llama_prompt = f"""Generate a music track structure in JSON format for: {prompt}

Return only valid JSON with this exact structure:

{{
  "BPM": 120,
  "structure": [
    {{"type": "intro", "duration": 8}},
    {{"type": "buildup", "duration": 16}},
    {{"type": "drop", "duration": 16}},
    {{"type": "outro", "duration": 8}}
  ]
}}

Rules:
- Only JSON, no explanations
- Use double quotes only
- Match the genre and mood from description
- Total duration should be 60-120 seconds
- Section types: intro, verse, chorus, buildup, drop, breakdown, bridge, outro"""

    models_to_try = [
        "llama3-music:latest",    # Ваша специализированная модель
        "llama3.1:latest",        # Стабильная версия
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
                input=llama_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=90  # Разумный таймаут
            )

            if result.returncode != 0:
                stderr_output = result.stderr.strip()
                if "model" in stderr_output.lower() and "not found" in stderr_output.lower():
                    logging.info(f"⚠️ Модель {model} не найдена, пропускаем")
                else:
                    logging.warning(f"⚠️ Ошибка ollama для {model}: {stderr_output}")
                continue

            raw_output = result.stdout.strip()
            
            if raw_output:
                logging.info(f"✅ Получен ответ от {model}: {len(raw_output)} символов")
                parsed_result = _parse_json_from_response(raw_output)
                if parsed_result:
                    return parsed_result

        except subprocess.TimeoutExpired:
            logging.warning(f"⚠️ Запрос к {model} превысил лимит времени")
            continue
        except FileNotFoundError:
            logging.error("❌ ollama команда не найдена. Установите Ollama!")
            break
        except Exception as e:
            logging.warning(f"⚠️ Неожиданная ошибка с {model}: {e}")
            continue

    logging.error("❌ Все попытки запроса к LLM через subprocess провалились")
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
        if stripped_line.startswith(('//:', '#', '/*', '*/', 'Note:', 'Explanation:', 'Human:', 'Assistant:')):
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
            # Логируем проблемный JSON для отладки
            logging.debug(f"Проблемный JSON: {candidate[:200]}")
            continue
        except Exception as e:
            logging.debug(f"⚠️ Ошибка обработки кандидата {i+1}: {e}")
            continue
    
    logging.error("❌ Не удалось найти валидный JSON в ответе LLM")
    logging.debug(f"Исходный ответ: {raw_output}")
    return None


def _clean_json_text(text: str) -> str:
    """
    Улучшенная очистка JSON текста от мусора
    """
    if not text:
        return ""
    
    # Убираем markdown разметку
    text = text.replace('```json', '').replace('```', '')
    
    # Убираем комментарии в стиле // и #
    lines = []
    for line in text.split('\n'):
        # Проверяем, не является ли строка комментарием
        stripped = line.strip()
        if not stripped.startswith(('//:', '//', '#', '/*', '*/', 'Note:', 'Explanation:')):
            # Убираем inline комментарии после //
            if '//' in line:
                line = line.split('//')[0]
            lines.append(line)
    
    text = '\n'.join(lines)
    
    # Убираем trailing commas (частая проблема LLM)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Убираем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*{\s*', '{', text)
    text = re.sub(r'\s*}\s*', '}', text)
    text = re.sub(r'\s*\[\s*', '[', text)
    text = re.sub(r'\s*\]\s*', ']', text)
    
    return text.strip()


def _validate_llama_response(data: dict) -> bool:
    """
    ИСПРАВЛЕННАЯ валидация структуры ответа от LLaMA
    """
    if not isinstance(data, dict):
        logging.debug("❌ Ответ не является словарем")
        return False
    
    # ИСПРАВЛЕНО: Проверяем BPM вместо tempo
    if 'BPM' not in data:
        logging.debug("❌ Нет поля BPM")
        return False
    
    if not isinstance(data['BPM'], (int, float)):
        logging.debug("❌ BPM не число")
        return False
    
    if 'structure' not in data or not isinstance(data['structure'], list):
        logging.debug("❌ Нет валидного поля structure")
        return False
    
    if not data['structure']:
        logging.debug("❌ Структура пуста")
        return False
    
    # Проверяем структуру секций
    for i, section in enumerate(data['structure']):
        if not isinstance(section, dict):
            logging.debug(f"❌ Секция {i} не является словарем")
            return False
        if 'type' not in section:
            logging.debug(f"❌ Секция {i} без типа")
            return False
        if 'duration' not in section:
            logging.debug(f"❌ Секция {i} без длительности")
            return False
        if not isinstance(section['duration'], (int, float)):
            logging.debug(f"❌ Секция {i} с некорректной длительностью")
            return False
    
    logging.debug("✅ Валидация структуры прошла успешно")
    return True


def _normalize_llama_response(data: dict) -> dict:
    """
    ИСПРАВЛЕННАЯ нормализация ответа от LLaMA к стандартному формату pipeline
    """
    # ИСПРАВЛЕНО: Возвращаем формат, который ожидает pipeline
    normalized = {
        'structure': [],
        'BPM': int(data.get('BPM', 120)),
        'tracks': data.get('tracks', [])  # Для совместимости
    }
    
    total_duration = 0
    
    # Нормализуем структуру
    for section in data.get('structure', []):
        duration = int(section.get('duration', 16))
        normalized_section = {
            'type': section.get('type', 'section').lower(),
            'duration': duration,
            'energy': _estimate_energy_from_type(section.get('type', 'section')),
            'start_time': total_duration  # Добавляем start_time для совместимости
        }
        normalized['structure'].append(normalized_section)
        total_duration += duration
    
    # Если структура пустая, создаем базовую
    if not normalized['structure']:
        logging.warning("⚠️ Создаем базовую структуру, так как LLM вернул пустую")
        normalized['structure'] = [
            {'type': 'intro', 'duration': 8, 'energy': 0.3, 'start_time': 0},
            {'type': 'verse', 'duration': 16, 'energy': 0.6, 'start_time': 8},
            {'type': 'chorus', 'duration': 16, 'energy': 0.8, 'start_time': 24},
            {'type': 'verse', 'duration': 16, 'energy': 0.6, 'start_time': 40},
            {'type': 'chorus', 'duration': 16, 'energy': 0.9, 'start_time': 56},
            {'type': 'outro', 'duration': 8, 'energy': 0.4, 'start_time': 72}
        ]
        total_duration = 80
    
    # Добавляем общую длительность для удобства
    normalized['total_duration'] = total_duration
    
    logging.info(f"✅ Структура нормализована: {len(normalized['structure'])} секций, {total_duration}с")
    return normalized


def _estimate_energy_from_type(section_type: str) -> float:
    """
    Оценка энергии секции по её типу
    """
    energy_map = {
        'intro': 0.3,
        'verse': 0.6,
        'prechorus': 0.7,
        'chorus': 0.8,
        'hook': 0.9,
        'drop': 0.9,
        'buildup': 0.7,
        'breakdown': 0.4,
        'bridge': 0.5,
        'outro': 0.3,
        'interlude': 0.4,
        'solo': 0.8
    }
    
    return energy_map.get(section_type.lower(), 0.5)


def test_mistral_client():
    """
    Тестирование клиента для отладки
    """
    print("🧪 Тестирование mistral_client...")
    
    test_prompts = [
        "dark trap beat 160bpm aggressive",
        "melodic lofi study music with vinyl textures",
        "energetic drum and bass 174bpm"
    ]
    
    for prompt in test_prompts:
        print(f"\n🎵 Тестируем: '{prompt}'")
        result = query_structured_music(prompt)
        
        if result:
            print(f"✅ Успех: {len(result['structure'])} секций, BPM: {result.get('BPM')}")
            for i, section in enumerate(result['structure']):
                print(f"  {i+1}. {section['type']} - {section['duration']}с")
        else:
            print("❌ Неудача")
    
    return True


if __name__ == "__main__":
    # Включаем отладочное логирование
    logging.basicConfig(level=logging.DEBUG)
    test_mistral_client()
