import cv2 as cv
from tqdm import tqdm
from Homography.HomographyDetector import HomographyDetector
from utils import merge_contours, combine_frames

cap = cv.VideoCapture("../Videos/slow_moves_plane_.mp4")
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# ----------------- Write to file -----------------
# fps = int(cap.get(cv.CAP_PROP_FPS))
# output_video_path = '../VideoResults/Homography/slow_moves_plane_.mp4'
# fourcc = cv.VideoWriter.fourcc(*'mp4v')
# out = cv.VideoWriter(output_video_path, fourcc, fps, (1240, 480))
# ----------------- ------------- -----------------

ret, frame = cap.read()
detector = HomographyDetector()
progress_bar = tqdm(total=total_frames, desc="Processing frames")

while cap.isOpened():
    ret, current_frame = cap.read()
    if not ret:
        break

    masked_frame = detector.detect(frame, current_frame, blur=True, blur_size=(4, 4), erode_dilate_iterations=3)
    if masked_frame is not None:
        contours, _ = cv.findContours(masked_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = merge_contours(contours)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        combined_frame = combine_frames(masked_frame, current_frame)
        cv.imshow('mask', combined_frame)
        frame = current_frame

        # ----------------- Write to file -----------------
        # resized = cv.resize(combined_frame, (1240, 480))
        # out.write(resized)
        # ----------------- ------------- -----------------

    key = cv.waitKey(30) & 0xFF
    if key == 27:
        break

    # Update progress bar
    progress_bar.update(1)

cap.release()
cv.destroyAllWindows()
progress_bar.close()
# out.release()
