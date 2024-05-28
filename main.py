import cv2

# Path to the video file
video_path = './VideoResults/WeightedWithSingleGaussian2/plane_moving_grass.mp4'

# Time in seconds from which to extract the frame
time_in_seconds = 25

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Calculate the frame number to extract
frame_number = int(fps * time_in_seconds)

# Set the video capture to the frame number
video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
success, frame = video_capture.read()

if success:
    # Save the frame as an image file
    cv2.imwrite('../CoursePhotos/plane_moving_grass/WeightedWithSingleGaussian2.png', frame)
    print('Frame extracted and saved as frame_at_6_seconds.jpg')
else:
    print('Failed to extract frame')

# Release the video capture object
video_capture.release()
