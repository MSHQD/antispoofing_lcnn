#!/usr/bin/env python3
"""
Скрипт для копирования модели в правильную папку.
Если у вас есть файл модели в другом месте, укажите путь ниже.
"""

import os
import shutil

def copy_model():
    """Копирует модель в правильную папку."""
    
    # Укажите здесь путь к вашему файлу модели
    source_model_path = input("Введите путь к файлу модели (например, /path/to/best_model.pth): ").strip()
    
    if not source_model_path:
        print("Путь не указан. Выход.")
        return
    
    if not os.path.exists(source_model_path):
        print(f"Файл {source_model_path} не найден.")
        return
    
    # Создаем папку назначения
    target_dir = "experiments/lcnn_asv_2"
    os.makedirs(target_dir, exist_ok=True)
    
    # Копируем файл
    target_path = os.path.join(target_dir, "best_model.pth")
    shutil.copy2(source_model_path, target_path)
    
    print(f"Модель скопирована в {target_path}")
    print("Теперь можно запустить inference_fixed.py")

if __name__ == "__main__":
    copy_model()