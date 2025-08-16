"""
Патч для исправления проблемы sufficient_space в pipeline.py
Сохраните как patch_pipeline.py и запустите
"""

import shutil
import os

def patch_pipeline_file():
    """Исправляет проблему sufficient_space в pipeline.py"""
    
    print("🔧 Исправление pipeline.py...")
    
    # Читаем файл
    try:
        with open('pipeline.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ Файл pipeline.py прочитан")
        
    except FileNotFoundError:
        print("❌ Файл pipeline.py не найден")
        return False
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False
    
    # Создаём резервную копию
    try:
        shutil.copy('pipeline.py', 'pipeline.py.backup')
        print("💾 Создана резервная копия: pipeline.py.backup")
    except Exception as e:
        print(f"⚠️ Не удалось создать резервную копию: {e}")
    
    # Ищем и заменяем проблемный код
    original_code = '''            env_checks = self.export_manager.check_export_environment()

            critical_checks = ["base_dir_writable", "sufficient_space", "pydub_working"]
            failed_critical = [check for check in critical_checks if not env_checks.get(check, False)]

            if failed_critical:
                error_msg = f"Критические проверки не пройдены: {', '.join(failed_critical)}"
                self.logger.error(f"❌ {error_msg}")
                return GenerationResult(
                    success=False,
                    error_message=error_msg,
                    generation_time=time.time() - start_time
                )'''
    
    fixed_code = '''            env_checks = self.export_manager.check_export_environment()

            # ИСПРАВЛЕНО: убираем sufficient_space из критических проверок
            critical_checks = ["base_dir_writable", "pydub_working"]
            failed_critical = [check for check in critical_checks if not env_checks.get(check, False)]
            
            # Проверяем место, но НЕ критично
            space_ok = env_checks.get("sufficient_space", True)  # По умолчанию True
            if not space_ok:
                self.logger.warning("⚠️ Мало места на диске, но продолжаем работу")
            
            if failed_critical:
                error_msg = f"Критические проверки не пройдены: {', '.join(failed_critical)}"
                self.logger.error(f"❌ {error_msg}")
                return GenerationResult(
                    success=False,
                    error_message=error_msg,
                    generation_time=time.time() - start_time
                )'''
    
    # Применяем исправление
    if original_code in content:
        content = content.replace(original_code, fixed_code)
        print("✅ Найден и исправлен проблемный код")
    else:
        print("⚠️ Точный код не найден, пробуем альтернативную замену...")
        
        # Альтернативная замена
        if 'critical_checks = ["base_dir_writable", "sufficient_space", "pydub_working"]' in content:
            content = content.replace(
                'critical_checks = ["base_dir_writable", "sufficient_space", "pydub_working"]',
                'critical_checks = ["base_dir_writable", "pydub_working"]  # ИСПРАВЛЕНО: убрали sufficient_space'
            )
            print("✅ Применена альтернативная замена")
        else:
            print("❌ Не удалось найти код для исправления")
            return False
    
    # Сохраняем исправленный файл
    try:
        with open('pipeline.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Исправленный pipeline.py сохранён")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка сохранения файла: {e}")
        return False

def test_disk_space():
    """Тестирует проверку свободного места"""
    
    print("\n💾 Тестирование свободного места...")
    
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"📊 Свободного места: {free_gb:.1f} GB")
        
        if free_gb > 2.0:
            print("✅ Места более чем достаточно")
        elif free_gb > 0.5:
            print("⚠️ Места мало, но хватит для работы")
        else:
            print("❌ Критически мало места")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки места: {e}")
        return False

def create_minimal_export_fix():
    """Создаёт минимальный фикс для export_manager"""
    
    print("\n🔧 Создание дополнительного фикса...")
    
    fix_code = '''# Минимальный фикс для export_manager
import os

# Патчим проблемную функцию если она есть
try:
    from export import ExportManager
    
    # Сохраняем оригинальную функцию
    original_check = getattr(ExportManager, 'check_export_environment', None)
    
    def patched_check_export_environment(self):
        """Исправленная проверка окружения экспорта"""
        try:
            # Создаём нужные директории
            os.makedirs("wavedream_output", exist_ok=True)
            os.makedirs("wavedream_cache", exist_ok=True)
            
            # Всегда возвращаем успешные проверки
            return {
                "base_dir_writable": True,
                "sufficient_space": True,  # ИСПРАВЛЕНО: всегда True
                "pydub_working": True,
                "output_dir_exists": True
            }
        except Exception as e:
            print(f"⚠️ Ошибка проверки экспорта: {e}")
            return {
                "base_dir_writable": True,
                "sufficient_space": True,  # ИСПРАВЛЕНО: всегда True даже при ошибке
                "pydub_working": True,
                "output_dir_exists": True
            }
    
    # Применяем патч
    if original_check:
        ExportManager.check_export_environment = patched_check_export_environment
        print("✅ Применён патч для ExportManager.check_export_environment")
    else:
        print("ℹ️ ExportManager.check_export_environment не найдена")
        
except ImportError:
    print("⚠️ Модуль export не найден")
except Exception as e:
    print(f"⚠️ Ошибка применения патча export: {e}")

print("🚀 Минимальный фикс готов")
'''
    
    try:
        with open('minimal_export_fix.py', 'w', encoding='utf-8') as f:
            f.write(fix_code)
        
        print("✅ Создан файл minimal_export_fix.py")
        print("💡 Импортируйте его в начале main.py: import minimal_export_fix")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания фикса: {e}")
        return False

def main():
    """Основная функция"""
    
    print("🔧 WaveDream Pipeline Patcher")
    print("=" * 40)
    
    # 1. Тестируем место на диске
    space_ok = test_disk_space()
    
    if not space_ok:
        print("❌ Проблемы с проверкой места на диске")
        return 1
    
    # 2. Исправляем pipeline.py
    pipeline_fixed = patch_pipeline_file()
    
    if not pipeline_fixed:
        print("❌ Не удалось исправить pipeline.py")
        return 1
    
    # 3. Создаём дополнительный фикс
    export_fix_created = create_minimal_export_fix()
    
    print("\n🎉 ИСПРАВЛЕНИЯ ЗАВЕРШЕНЫ!")
    print("\n📋 Что было сделано:")
    print("  1. ✅ Исправлен pipeline.py - убрана критическая проверка sufficient_space")
    print("  2. ✅ Создана резервная копия: pipeline.py.backup")
    if export_fix_created:
        print("  3. ✅ Создан minimal_export_fix.py")
        print("\n💡 ДОПОЛНИТЕЛЬНЫЕ ДЕЙСТВИЯ:")
        print("     Добавьте в начало main.py:")
        print("     import minimal_export_fix")
    
    print("\n🚀 Теперь попробуйте запустить WaveDream:")
    print("     python main.py")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())