import numpy as np
import cv2

# ------------------------- Contours -------------------------
def calculate_contour_distance(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

def take_biggest_contours(contours, max_number):
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return sorted_contours[:max_number]

# @timed
def agglomerative_cluster(contours, threshold_distance):
    current_contours = list(contours)
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            # merge closest two contours
            index1, index2 = min_coordinate
            # current_contours[index1] = np.concatenate((current_contours[index1], current_contours[index2]), axis=0)
            current_contours[index1] = np.concatenate(
                (current_contours[index1].tolist(), current_contours[index2].tolist()), axis=0)

            del current_contours[index2]
        else:
            break
    return current_contours

def merge_contours(contours, max_contours_number=30, threshold_distance=40):
    contours = take_biggest_contours(contours, max_contours_number)
    contours = agglomerative_cluster(contours, threshold_distance)
    return contours

# ------------------------- Combine Frames -------------------------
def combine_frames(frame1, frame2):
    # Resize frames to have the same dimensions
    if len(frame1.shape) == 2:  # Grayscale frame
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    if len(frame2.shape) == 2:  # Grayscale frame
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    frame1_resized = cv2.resize(frame1, (640, 480))
    frame2_resized = cv2.resize(frame2, (640, 480))

    combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

    return combined_frame
