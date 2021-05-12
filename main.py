import cv2
import pafy
import numpy as np
import matplotlib.pyplot as plt

# видос с ютуба
video = pafy.new('https://www.youtube.com/watch?v=VeFl0LPzjYU')
best = video.getbest(preftype='mp4')
playurl = best.url

# захватывать кадры из видео
# cap = cv2.VideoCapture('video.avi')
cap = cv2.VideoCapture(playurl)

# Обученные классификаторы XML описывают некоторые особенности некоторого объекта, который мы хотим обнаружить
car_cascade = cv2.CascadeClassifier('cars.xml')
hsv_min = np.array((53, 0, 0), np.uint8)
hsv_max = np.array((83, 255, 255), np.uint8)

test_one = ''

# цикл запускается, если захват был инициализирован.
while True:

    # читает кадры из видео
    ret, frames = cap.read()

    # конвертировать в оттенки серого каждого кадра
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)



    if test_one != '':
        # gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        grayA = cv2.cvtColor(test_one, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        diff_image = cv2.absdiff(grayB, grayA)

        # perform image thresholding
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        # plot image after thresholding
        plt.imshow(dilated, cmap='gray')
        plt.imshow(dilated)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_cntrs = []
        for i, cntr in enumerate(contours):
            valid_cntrs.append(cntr)
        cv2.drawContours(frames, valid_cntrs, -1, (127, 200, 0), 2)
        test_one = ''

    else:
        test_one = frames





    # Обнаруживает автомобили разных размеров на входном изображении
    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    #
    # # Нарисовать прямоугольник в каждом авто
    # print(f'Авто в кадре: {len(cars)}')
    # for (x, y, w, h) in cars:
    #     # print(w, h)
    #     if 120 > w > 70 and 120 > h > 70:
    #         cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)



    # Отображать кадры в окне
    # cv2.imshow('video', frames)
    cv2.imshow('video', frames)
    # Дождаться остановки клавиши Esc
    if cv2.waitKey(33) == 27:
        break

# Отменить выделение любого связанного использования памяти
cv2.destroyAllWindows()
