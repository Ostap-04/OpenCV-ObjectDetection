"""
HomographyDetector - використовуємо гомографію для створення панорами, щоб компенсувати рух
Пробували використовувати два методи для порівняння дескрипторів: FLANN і Brute-Force. Зупинилися на Brute-Force
Для виявлення та опису ключових точок використовуємо ORB (Oriented FAST and Rotated BRIEF).

Метод detect:
Використовуємо метод find_panorama для створення панорами з поточного кадру і попереднього.
Якщо попередній трансформований кадр доступний, обчислюємо різницю між поточним і попереднім трансформованими кадрами.
Використовуємо threshold, щоб виділити зміни між кадрами.
Застосовуємо blur та erode/dilate для погашення шумів.
Оновлюємо попередній трансформований кадр для наступної ітерації.
"""
import numpy as np
import cv2 as cv

class HomographyDetector():
    def __init__(self):
        self.FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        self.matcher = cv.BFMatcher(crossCheck=True, normType=cv.NORM_HAMMING)
        self.orb = cv.ORB.create()
        self.prev_transformed = None
        self.kernel = np.ones((3, 3), np.uint8)
        self.prev_d = None
        self.prev_kp = None

    def detectFlann(self, prev_frame, current_frame):
        prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        [keypoints, descriptors] = self.orb.detectAndCompute(current_frame_gray, None)
        if descriptors is not None:
            if prev_frame is not None and self.prev_d is not None:
                matches = self.flann.knnMatch(self.prev_d.astype(np.float32), descriptors.astype(np.float32), k=2)
                good_matches = []
                for m, n in matches:
                    if abs(m.distance - n.distance) > 0:
                        good_matches.append(m)

                src_points = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dist_points = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                if len(src_points) >= 4 and len(dist_points) >= 4:
                    hm_matrix, mask = cv.findHomography(src_points, dist_points, cv.RANSAC)
                    if hm_matrix is not None:
                        transformed = cv.warpPerspective(prev_frame_gray, hm_matrix,
                                                              (prev_frame_gray.shape[1], prev_frame_gray.shape[0]))
                        self.prev_d = descriptors
                        self.prev_kp = keypoints
                        return transformed

            self.prev_d = descriptors
            self.prev_kp = keypoints
        return None

    def find_panorama(self, prev_frame, current_frame):
        prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        prev_frame_kp, prev_frame_descriptors = self.orb.detectAndCompute(prev_frame_gray, None)
        current_frame_kp, current_frame_descriptors = self.orb.detectAndCompute(current_frame_gray, None)

        matches = self.matcher.match(prev_frame_descriptors, current_frame_descriptors)
        # matches = sorted(matches, key=lambda x: x.distance)
        matched_keypoints_previous = np.float32([prev_frame_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        matched_keypoints_current = np.float32([current_frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(matched_keypoints_previous) >= 4 and len(matched_keypoints_current) >= 4:
            homography_matrix, mask = cv.findHomography(matched_keypoints_current,
                                                        matched_keypoints_previous,
                                                        cv.RANSAC, 5.0)
            transformed = cv.warpPerspective(current_frame_gray, homography_matrix, (prev_frame_gray.shape[1], prev_frame_gray.shape[0]))

            return transformed

        return None

    def detect(self, prev_frame, current_frame, thresh=30, blur=False, blur_size=(3, 3), erode_dilate_iterations=0):
        transformed = self.find_panorama(prev_frame, current_frame)
        res = None
        if self.prev_transformed is not None:
            frame_diff = cv.absdiff(transformed, self.prev_transformed)
            _, res = cv.threshold(frame_diff, thresh, 255, cv.THRESH_BINARY)
            if blur:
                res = cv.blur(res, blur_size)

            res = cv.erode(res, self.kernel, iterations=erode_dilate_iterations)
            res = cv.dilate(res, self.kernel, iterations=erode_dilate_iterations)
        self.prev_transformed = transformed

        return res
