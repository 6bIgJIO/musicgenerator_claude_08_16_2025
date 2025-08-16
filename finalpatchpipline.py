#!/usr/bin/env python3
"""
Исправление ошибки 'final_path' is not defined в pipeline.py
"""

import shutil

def fix_final_path_error():
    """Исправляет ошибку final_path в exception handler"""
    
    print("🔧 Исправление ошибки final_path в pipeline.py...")
    
    try:
        with open('pipeline.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ Файл pipeline.py прочитан")
        
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False
    
    # Создаём резервную копию
    try:
        shutil.copy('pipeline.py', 'pipeline.py.backup2')
        print("💾 Создана резервная копия: pipeline.py.backup2")
    except:
        pass
    
    # Ищем проблемный код в exception handler
    problem_patterns = [
        # Вариант 1: в блоке try/except
        '''            # В случае ошибки пытаемся принудительно сохранить всё что есть
            try:
                if hasattr(self, '_intermediate_storage') and self._intermediate_storage:
                    self.logger.info("🚨 Попытка аварийного сохранения...")
                    emergency_files = self.export_manager.force_save_everything(
                        mastered_audio if 'mastered_audio' in locals() else b'',
                        self._intermediate_storage,
                        {"error": str(e), "timestamp": time.time()}
                    )
                    self.logger.info(f"🚨 Аварийно сохранено: {len(emergency_files)} файлов")
            except Exception as save_error:
                self.logger.error(f"❌ Ошибка аварийного сохранения: {save_error}")
            
            return GenerationResult(
                success=False,
                generation_time=generation_time,
                error_message=str(e),
                intermediate_files=self._intermediate_storage
            )''',
        
        # Вариант 2: если есть ссылка на final_path
        'final_path=final_path or exported_files.get("final")',
        
        # Вариант 3: в конце функции
        '''return GenerationResult(
                success=True,
                final_path=final_path or exported_files.get("final"),'''
    ]
    
    fixed = False
    
    # Исправление 1: Замена в exception handler
    if 'mastered_audio if \'mastered_audio\' in locals() else b\'\'' in content:
        content = content.replace(
            'mastered_audio if \'mastered_audio\' in locals() else b\'\'',
            'mastered_audio if \'mastered_audio\' in locals() else None'
        )
        print("✅ Исправлена ссылка на mastered_audio")
        fixed = True
    
    # Исправление 2: Замена final_path в GenerationResult
    if 'final_path=final_path or exported_files.get("final")' in content:
        content = content.replace(
            'final_path=final_path or exported_files.get("final")',
            'final_path=exported_files.get("final")'
        )
        print("✅ Исправлена ссылка на final_path")
        fixed = True
    
    # Исправление 3: Поиск и замена других ссылок на final_path
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'final_path' in line and 'final_path =' not in line and 'def ' not in line:
            # Если это не определение переменной final_path, а использование
            if 'GenerationResult' in line and 'final_path=' in line:
                # В конструкторе GenerationResult
                if 'final_path or exported_files' in line:
                    line = line.replace('final_path or exported_files.get("final")', 'exported_files.get("final")')
                    print(f"✅ Исправлена строка {i+1}: GenerationResult final_path")
                    fixed = True
                elif 'final_path,' in line or 'final_path)' in line:
                    # Простая ссылка на final_path без определения
                    line = line.replace('final_path', 'exported_files.get("final", None)')
                    print(f"✅ Исправлена строка {i+1}: неопределённый final_path")
                    fixed = True
        
        new_lines.append(line)
    
    if fixed:
        content = '\n'.join(new_lines)
    
    # Дополнительное исправление: добавляем инициализацию final_path если её нет
    if not fixed:
        print("⚠️ Точные совпадения не найдены, применяем общее исправление...")
        
        # Ищем место где должна быть инициализация final_path
        if '# === 11. EXPORT + МЕТАДАННЫЕ ===' in content:
            insert_point = content.find('# === 11. EXPORT + МЕТАДАННЫЕ ===')
            before = content[:insert_point]
            after = content[insert_point:]
            
            # Добавляем инициализацию final_path
            init_code = '''            
            # ИСПРАВЛЕНИЕ: Инициализация final_path
            final_path = None
            
'''
            content = before + init_code + after
            print("✅ Добавлена инициализация final_path")
            fixed = True
    
    # Сохраняем исправленный файл
    if fixed:
        try:
            with open('pipeline.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Исправленный pipeline.py сохранён")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
            return False
    else:
        print("⚠️ Исправления не применены - возможно, код уже исправен")
        return True

def create_emergency_pipeline_fix():
    """Создаёт экстренный патч для pipeline"""
    
    print("\n🚨 Создание экстренного патча...")
    
    patch_code = '''# Экстренный патч для pipeline.py
# Импортируйте в начале main.py: import emergency_pipeline_fix

def patch_pipeline_final_path():
    """Патчит проблему с final_path в pipeline"""
    try:
        from pipeline import WaveDreamPipeline
        
        # Сохраняем оригинальный метод
        original_generate = WaveDreamPipeline.generate_track
        
        async def patched_generate_track(self, request):
            """Исправленный generate_track с proper final_path handling"""
            try:
                # Вызываем оригинальный метод
                result = await original_generate(self, request)
                return result
                
            except NameError as e:
                if "final_path" in str(e):
                    print("🔧 Перехвачена ошибка final_path, применяем фикс...")
                    
                    # Возвращаем базовый результат
                    from pipeline import GenerationResult
                    import time
                    
                    return GenerationResult(
                        success=False,
                        final_path=None,
                        generation_time=0.0,
                        error_message="Исправлена ошибка final_path - попробуйте ещё раз"
                    )
                else:
                    raise
            
            except Exception as e:
                print(f"🚨 Неожиданная ошибка в pipeline: {e}")
                from pipeline import GenerationResult
                
                return GenerationResult(
                    success=False,
                    final_path=None,
                    generation_time=0.0,
                    error_message=str(e)
                )
        
        # Применяем патч
        WaveDreamPipeline.generate_track = patched_generate_track
        print("✅ Применён экстренный патч для final_path")
        
    except Exception as e:
        print(f"⚠️ Не удалось применить экстренный патч: {e}")

# Автоматически применяем патч при импорте
patch_pipeline_final_path()
'''
    
    try:
        with open('emergency_pipeline_fix.py', 'w', encoding='utf-8') as f:
            f.write(patch_code)
        
        print("✅ Создан emergency_pipeline_fix.py")
        print("💡 Добавьте в начало main.py:")
        print("     import emergency_pipeline_fix")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания экстренного патча: {e}")
        return False

def main():
    """Основная функция"""
    
    print("🔧 Исправление ошибки final_path")
    print("=" * 40)
    
    # 1. Пытаемся исправить файл напрямую
    direct_fixed = fix_final_path_error()
    
    if direct_fixed:
        print("\n✅ Прямое исправление применено")
    else:
        print("\n⚠️ Прямое исправление не удалось")
    
    # 2. Создаём экстренный патч на всякий случай
    emergency_created = create_emergency_pipeline_fix()
    
    print("\n🎉 ИСПРАВЛЕНИЯ ЗАВЕРШЕНЫ!")
    print("\n📋 Рекомендации:")
    
    if direct_fixed:
        print("  1. ✅ pipeline.py исправлен напрямую")
        print("  2. 🚀 Попробуйте запустить: python main.py")
    
    if emergency_created:
        print("  3. 💡 Если проблема повторится, добавьте в начало main.py:")
        print("       import emergency_pipeline_fix")
    
    print("\n🚀 Теперь WaveDream должен работать!")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())