import cv2

# Define the blur radius and kernel sizes for erosion and dilation
BLUR_RADIUS = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Initialize a video capture
cap = cv2.VideoCapture(0)

# Open the video file for reading
#cap = cv2.VideoCapture("videos/pedestrians.avi")

# Capture 10 frames to allow the camera's autoexposure to adjust.
for i in range(10):
    success, frame = cap.read()
if not success:
    exit(1)

# Convert the 10th frame to grayscale and apply Gaussian blur
gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_background = cv2.GaussianBlur(gray_background,(BLUR_RADIUS, BLUR_RADIUS), 0)

# Capture a frame from the camera
success, frame = cap.read()
while success:

    # Convert the current frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(BLUR_RADIUS, BLUR_RADIUS), 0)

    # Calculate the absolute difference between the background and the current frame
    diff = cv2.absdiff(gray_background, gray_frame)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # Apply morphological erosion and dilation to smoothen the thresholded image
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    # Find contours of objects in the thresholded image
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected contours and draw bounding rectangles for large ones
    for c in contours:
        if cv2.contourArea(c) > 4000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Diff', diff)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Detection', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    # Capture a frame from the camera
    success, frame = cap.read()
