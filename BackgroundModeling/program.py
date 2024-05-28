import cv2
import numpy as np
from SingleGaussianModel import SingleGaussianModel as SGM
from SingleGaussian2 import SGM as SGM2
import WeightedModel as WM
from Homography.HomographyDetector import HomographyDetector
import utils
from tqdm import tqdm

def get_corner_pixels(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    top_left_corner = frame[0, 0]
    top_right_corner = frame[0, width - 1]
    bottom_left_corner = frame[height - 1, 0]
    bottom_right_corner = frame[height - 1, width - 1]
    corner_pixels = np.array([
        top_left_corner,
        top_right_corner,
        bottom_left_corner,
        bottom_right_corner
    ])
    return corner_pixels

cap = cv2.VideoCapture("../Videos/plane_city.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# ----------------- Write to file -----------------
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video_path = '../VideoResults/ImprovedSGM/SGM2.mp4'
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1240, 480))
# ----------------- ------------- -----------------

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame,(1240, 480))
detector = HomographyDetector()
progress_bar = tqdm(total=total_frames, desc="Processing frames")

gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# short_model = SGM(initial_frame=prev_frame, learning_rate=0.06, decision_threshold=2.5, blur=True)
# short_model = cv2.createBackgroundSubtractorMOG2(varThreshold=250, detectShadows=False)
short_model = SGM2(prev_frame, learning_rate=0.04, age_cap=8, diff_threshold=5, blur=True, blur_ksize=(3, 3), gaussian_blur=False, blur_intensity=1.5)
long_model = WM.WeightedModel(initial_frame=gray_frame, capture_frequency=2, frames_capture_num=4, learning_rate=0.4, blur=True, blur_size=(3, 3))

luminosity_change_threshold_in_corner_pixels = 5
corners = get_corner_pixels(prev_frame)
prev_time = 0
resized = None
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1240, 480))
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_curr = get_corner_pixels(frame)
    luminosity_change = np.abs(np.average(corners_curr) - np.average(corners))
    changed = luminosity_change > luminosity_change_threshold_in_corner_pixels
    corners = corners_curr
    homography = None
    input = gray

    # if np.all(changed):
    #     print("changed")
    #     homography = detector.find_panorama(prev_frame, frame)
    #     input = homography

    # long_model_mask = long_model.apply(input)
    short_model_mask = short_model.apply(input)
    result = short_model_mask
    # if long_model_mask is not None:
    #     result = cv2.bitwise_and(short_model_mask, long_model_mask)
    #
    #     contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = utils.merge_contours(contours, 40, 35)
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     combined_frames = utils.combine_frames(result, frame)
    #     resized = cv2.resize(combined_frames, (1240, 480))
    #     cv2.imshow('Frame', resized)
    #     # out.write(resized)
    # elif resized is not None:
    #     cv2.imshow('Frame', resized)
    #     # out.write(resized)


    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = utils.merge_contours(contours, 40, 35)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    combined_frames = utils.combine_frames(result, frame)
    # ----------------- Write to file -----------------
    resized = cv2.resize(combined_frames, (1240, 480))
    cv2.imshow('Frame', resized)

    out.write(resized)
    # ----------------- ------------- -----------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    progress_bar.update(1)

cap.release()
cv2.destroyAllWindows()
progress_bar.close()
# out.release()

