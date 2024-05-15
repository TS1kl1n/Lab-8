import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'images/variant-7.jpg'
image = cv2.imread(image_path)

# Проверка загрузки изображения
if image is None:
    raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

# Отразить по горизонтали
flipped_image = cv2.flip(image, 1)

# Перевернуть
flipped_and_rotated_image = cv2.flip(flipped_image, 0)

# Сохранение преобразованного изображения
output_path = 'images/variant-7-transformed.jpg'
cv2.imwrite(output_path, flipped_and_rotated_image)

# Отображение исходного и преобразованного изображений
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Преобразованное изображение')
plt.imshow(cv2.cvtColor(flipped_and_rotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

import cv2
import numpy as np

def find_marker(image):
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение гауссового размытия
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Применение порогового значения
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Выбор самого большого контура
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

def draw_distance_to_center(image, marker_contour):
    moments = cv2.moments(marker_contour)
    if moments['m00'] == 0:
        return
    
    # Координаты центра метки
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    
    # Координаты центра кадра
    frame_center_x = image.shape[1] // 2
    frame_center_y = image.shape[0] // 2
    
    # Вычисление расстояния до центра кадра
    distance_x = center_x - frame_center_x
    distance_y = center_y - frame_center_y
    
    # Отрисовка центра метки
    cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)
    # Отрисовка центра кадра
    cv2.circle(image, (frame_center_x, frame_center_y), 10, (255, 0, 0), -1)
    
    # Отображение расстояния
    text = f'Distance to center: ({distance_x}, {distance_y}) px'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Захват видео с камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Не удалось открыть камеру")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    marker_contour = find_marker(frame)
    if marker_contour is not None:
        draw_distance_to_center(frame, marker_contour)
    
    # Отображение кадра
    cv2.imshow('Tracking', frame)
    
    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
