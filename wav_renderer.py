import pyautogui
import time

def render_wav():
    """
    Черновой метод для запуска рендера WAV через FL Studio.
    Обычно в FL Studio - Ctrl+R - Render dialog
    Потом Enter - подтверждение
    """
    print("[INFO] Запускаем рендер WAV...")
    pyautogui.hotkey('ctrl', 'r')
    time.sleep(2)
    pyautogui.press('enter')
    time.sleep(10)  # ждём рендера (нужно настроить под длину трека)

    print("[OK] Рендер WAV завершён.")