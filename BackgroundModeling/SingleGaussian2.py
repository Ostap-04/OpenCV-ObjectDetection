
import cv2
import numpy as np
import utils
from tqdm import tqdm
# import WeightedModel as WM
import cv2

class SGM():
    def __init__(self, initial_frame, learning_rate=0.1, age_cap=8, diff_threshold=5, blur=False, blur_ksize=(3, 3),
                 gaussian_blur=False, blur_intensity=1.5):
        self.learning_rate = learning_rate
        self.age_cap = age_cap
        self.diff_threshold = diff_threshold  # Threshold for difference to update alpha
        self.blur = blur
        self.blur_ksize = blur_ksize
        self.gaussian_blur = gaussian_blur
        self.blur_intensity = blur_intensity
        self.mu = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.sigma2 = np.ones_like(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32) * 15
        self.alpha = np.ones_like(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32)

    def update_background_model(self, frame):
        # Calculate M_t (current pixel value)
        M_t = frame

        # Calculate the absolute difference
        diff = np.abs(frame - self.mu)

        # Update age only if the difference is below the threshold
        alpha_increase = 1
        alpha_update_condition = diff < self.diff_threshold
        self.alpha = np.where(alpha_update_condition, np.minimum(self.alpha + alpha_increase, self.age_cap), self.alpha)

        # Update age
        # alpha = np.minimum(alpha + alpha_increase, age_cap)

        rho = self.learning_rate / self.alpha

        # Update mean (mu)
        self.mu = (1 - rho) * self.mu + rho * M_t
        # self.mu = (self.alpha/(1+self.alpha)) * self.mu + 1/(1+self.alpha) * M_t


        # Update variance (sigma^2)
        V_t = (self.mu - M_t) ** 2
        self.sigma2 = (1 - rho) * self.sigma2 + rho * V_t
        # self.sigma2 = self.alpha/(1+self.alpha) * self.sigma2 + 1/(1+self.alpha) * V_t

        return M_t

    def apply(self, frame):
        if self.blur:
            frame = cv2.blur(frame, self.blur_ksize)
        if self.gaussian_blur:
            frame = cv2.GaussianBlur(frame, self.blur_ksize, self.blur_intensity)
        frame = frame.astype(np.float32)
        self.update_background_model(frame)
        diff = np.abs(frame - self.mu)
        foreground_mask = diff > 2.5 * np.sqrt(self.sigma2)
        return foreground_mask.astype(np.uint8) * 255

def runSGM():
    cap = cv2.VideoCapture("../Videos/plane_moving_grass.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    sgm = SGM(prev_frame, learning_rate=0.03, age_cap=8, diff_threshold=5, blur=True, blur_ksize=(3, 3),
                 gaussian_blur=False, blur_intensity=1.5)

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

# runSGM()
BackgroundModeling/
        Formulas/
        Homography/
        HomographyAlgorithmPhotos/
        VideoResults/
        Videos/
        __pycache__/
        main.py
        utils.py
