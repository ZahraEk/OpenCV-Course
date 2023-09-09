import cv2

# Create a background subtractor using the GMG method with shadow detection enabled.
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Define the  kernel sizes for erosion and dilation
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))

# Open the video file for reading
cap = cv2.VideoCapture("videos/traffic.flv")

# Define the output video file name and codec
output_file = "videos/gmg_traffic.flv"
fourcc = cv2.VideoWriter_fourcc(*'FLV1')

# Get the frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a VideoWriter object to save the video
out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# Capture a frame from the camera
success, frame = cap.read()
while success:

    # Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

    # Apply morphological erosion and dilation to smoothen the thresholded image
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    # Find contours of objects in the thresholded image
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected contours and draw bounding rectangles for large ones
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    #Show the output videos
    cv2.imshow('GMG', fg_mask)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Detection', frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break

    # Capture a frame from the camera
    success, frame = cap.read()

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()