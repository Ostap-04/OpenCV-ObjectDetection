"""
Алгоритм SingleGaussianModel призначений для виявлення рухомих об'єктів у відео шляхом моделювання кожного пікселя фону
за допомогою Гауссового розподілу.

Зберігаємо початковий кадр як початкове середнє значення (self.mean).
Ініціалізуємо матрицю варіанс (self.var) одиничними значеннями.
Встановлюємо початкові значення для моделі фону (self.model), learning_rate, decision_threshold,
параметрів розмиття (якщо використовується)

Обчислюємо нове середнє значення (new_mean) для кожного пікселя за допомогою поточного кадру і
попереднього середнього значення з використанням learning_rate.
Обчислюємо нову варіансу (new_var) для кожного пікселя, враховуючи відхилення поточного кадру від попереднього
середнього значення.
Обчислюємо значенння для класифікації кожного пікселя як відношення квадрату різниці між поточним кадром і с
ереднім значенням до варіанси.
Оновлюємо середнє значення і дисперсію для наступного кадру.
Алгоритм SingleGaussianModel використовує модель одного Гауссового розподілу для кожного пікселя,
щоб відстежувати зміни у відео. Він виявляє рухомі об'єкти, порівнюючи поточний кадр з моделлю фону та
визначає, чи відхиляється піксель від очікуваного значення більше, ніж на заданий поріг.
(Formulas/SGM-*)
"""

import cv2
import numpy as np
import utils
from tqdm import tqdm

class SingleGaussianModel():
    def __init__(self, initial_frame, learning_rate=0.04, decision_threshold=2.5, blur=False, blur_size=(4, 4)):
        self.model = None
        self.learning_rate = learning_rate
        self.frame_counter = 0
        self.blur = blur
        self.blur_size = blur_size
        self.mean = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        (self.col, self.row) = self.mean.shape
        self.var = np.ones((self.col, self.row))
        self.decision_threshold = decision_threshold
        self.model = np.zeros(initial_frame.shape)

    def apply(self, frame):
        if self.blur:
            frame = cv2.blur(frame, self.blur_size)
        new_mean = (1 - self.learning_rate) * self.mean + self.learning_rate * frame
        new_mean = new_mean.astype(np.uint8)
        new_var = (1 - self.learning_rate) * self.var + self.learning_rate * (frame - self.mean) ** 2
        decision = np.divide((frame - self.mean) ** 2, self.var)
        frame_copy = frame.copy()
        frame_copy[:] = np.where(decision > self.decision_threshold, 255, 0)
        self.mean = new_mean
        self.var = new_var

        self.frame_counter += 1
        return frame_copy

# ------------------ Run SingleGaussianModel ------------------
def runSingleGaussianModel():
    cap = cv2.VideoCapture("../Videos/foggy_mountain.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    sgm = SingleGaussianModel(initial_frame=prev_frame, learning_rate=0.06, decision_threshold=3.5, blur=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgm_mask = sgm.apply(gray)

        contours, _ = cv2.findContours(sgm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = utils.merge_contours(contours, 40, 40)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        combined_frames = utils.combine_frames(sgm_mask, frame)
        resized = cv2.resize(combined_frames, (1240, 620))

        cv2.imshow('Frame', resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        progress_bar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()

# runSingleGaussianModel()
