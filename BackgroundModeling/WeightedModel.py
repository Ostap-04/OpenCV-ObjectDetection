"""
Алгоритм WeightedModel призначений для виявлення рухомих об'єктів у відео шляхом побудови моделі фону з використанням
зваженого середнього значення.

Ініціалізація:

Встановлюємо частоту захоплення кадрів (capture_frequency), кількість кадрів для оновлення моделі (frames_capture_num),
learning_rate, параметри розмиття (якщо використовується).

Якщо лічильник кадрів кратний capture_frequency, оновлюємо модель фону зваженим середнім поточного кадру
(Formulas/accumulateWeighted) з коефіцієнтом learning_rate.
Якщо кількість захоплень досягла frames_capture_num, то обчислюємо різницю між поточним кадром і моделлю фону.
Виконує threshold для створення маски руху, яку потім повертає як результат.
Алгоритм WeightedModel будує та постійно оновлює модель фону, дозволяючи виділяти рухомі об'єкти шляхом
аналізу різниці між поточним кадром і моделлю фону.
"""

import cv2
import numpy as np
import utils
from tqdm import tqdm

class WeightedModel():
    def __init__(self, initial_frame, capture_frequency=5, frames_capture_num=10, learning_rate=0.04, blur=False, blur_size=(4,4)):
        self.model = None
        self.capture_frequency = capture_frequency
        self.frames_capture_num = frames_capture_num
        self.learning_rate = learning_rate
        self.frame_counter = 0
        self.blur = blur
        self.blur_size = blur_size
        self.prev = initial_frame
        self.capture_count = 0

    def apply(self, _frame):
        frame = _frame.copy()
        if self.blur:
            frame = cv2.blur(frame, self.blur_size)
        if self.model is None or self.frame_counter == 0:
            self.model = frame.copy().astype(np.float32)
        elif self.frame_counter % self.capture_frequency == 0:
            cv2.accumulateWeighted(frame, self.model, alpha=self.learning_rate)
            self.capture_count += 1

        self.frame_counter += 1

        if self.capture_count >= self.frames_capture_num:
            mean_frame_uint8 = cv2.convertScaleAbs(self.model)
            mask = cv2.absdiff(mean_frame_uint8, frame)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
            self.capture_count = 0
            return mask


# ------------------ Run WeightedModel ------------------
def runWeightedModel():
    cap = cv2.VideoCapture("../Videos/plane_moving_grass.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    wm = WeightedModel(initial_frame=prev_frame, capture_frequency=1, frames_capture_num=1, learning_rate=0.4,
                       blur=True, blur_size=(3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wm_mask = wm.apply(gray)
        if wm_mask is not None:
            contours, _ = cv2.findContours(wm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = utils.merge_contours(contours, 40, 40)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            combined_frames = utils.combine_frames(wm_mask, frame)
            resized = cv2.resize(combined_frames, (1240, 620))
            cv2.imshow('Frame', resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        progress_bar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()

# runWeightedModel()
