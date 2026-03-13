import cv2
import sys
import os

def resize_video(input_path, output_path, width, height):
    """
    Изменяет размер видео до указанных ширины и высоты
    
    Args:
        input_path (str): путь к исходному видео
        output_path (str): путь для сохранения результата
        width (int): новая ширина
        height (int): новая высота
    """
    # Открываем видео
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {input_path}")
        return False
    
    # Получаем информацию о видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Исходное видео: {original_width}x{original_height}, {fps} fps, {total_frames} кадров")
    print(f"Новый размер: {width}x{height}")
    
    # Определяем кодек и создаем объект для записи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'avc1' для macOS, 'XVID' для AVI
    
    # Проверяем расширение выходного файла
    if output_path.lower().endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Ошибка: Не удалось создать выходной файл")
        cap.release()
        return False
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Изменяем размер кадра
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Записываем кадр
        out.write(resized_frame)
        
        frame_count += 1
        
        # Показываем прогресс каждые 100 кадров
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Прогресс: {frame_count}/{total_frames} кадров ({progress:.1f}%)")
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Готово! Обработано {frame_count} кадров")
    print(f"Результат сохранён в {output_path}")
    return True

def parse_size(size_str):
    """
    Парсит строку с размером вида "ШИРИНА:ВЫСОТА" или "ШИРИНАxВЫСОТА"
    
    Args:
        size_str (str): строка с размером
    
    Returns:
        tuple: (ширина, высота) или None при ошибке
    """
    try:
        # Поддерживаем форматы "1280:720" и "1280x720"
        if ':' in size_str:
            w, h = size_str.split(':')
        elif 'x' in size_str or 'х' in size_str:  # латинская и русская 'x'
            w, h = size_str.replace('х', 'x').split('x')
        else:
            return None
        
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None

def main():
    # Проверяем аргументы командной строки
    if len(sys.argv) != 3:
        print("Использование: python resize_video.py <путь_к_видео> <ширина:высота>")
        print("Пример: python resize_video.py video.mp4 1280:720")
        print("Пример: python resize_video.py video.mp4 640x480")
        return
    
    input_path = sys.argv[1]
    size_str = sys.argv[2]
    
    # Парсим размер
    size = parse_size(size_str)
    if size is None:
        print("Ошибка: Неправильный формат размера. Используйте ШИРИНА:ВЫСОТА или ШИРИНАxВЫСОТА")
        return
    
    width, height = size
    
    # Проверяем существование входного файла
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден")
        return
    
    # Генерируем имя выходного файла
    file_dir = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    name_without_ext, ext = os.path.splitext(file_name)
    output_path = os.path.join(file_dir, f"{name_without_ext}_resized{ext}")
    
    # Если выходной файл уже существует, добавляем номер
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(file_dir, f"{name_without_ext}_resized_{counter}{ext}")
        counter += 1
    
    # Изменяем размер видео
    resize_video(input_path, output_path, width, height)

if __name__ == "__main__":
    main()
