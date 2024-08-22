import cv2
import os
import random
import numpy as np

# Directory to save the processed face images
prepared_dir = "PATH_TO_Face_images_folder"

# Initialize the face detector using a pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video source (0 = default camera, can be changed if needed)
vid = 0

# Ensure that the directory for processed face images exists
if not os.path.exists(prepared_dir):
    os.makedirs(prepared_dir)

# Start video capture from the specified video source
cap = cv2.VideoCapture(vid)
count = 0  # Counter for saved face images

# Function to detect if an image is blurry
def is_blurry(image, threshold=30.0):
    # Convert image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of the Laplacian, a measure of sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If the variance is below the threshold, the image is considered blurry
    return laplacian_var < threshold

# Function to check if the full head is within the frame
def is_full_head_in_frame(face, frame, min_face_size_ratio=0.3, max_face_size_ratio=0.9):
    x, y, w, h = face  # Coordinates and size of the detected face
    frame_h, frame_w = frame.shape[:2]  # Height and width of the video frame
    
    # Check if the face size is within the acceptable range as a ratio of the frame size
    face_size_ratio = max(w / frame_w, h / frame_h)
    if face_size_ratio < min_face_size_ratio or face_size_ratio > max_face_size_ratio:
        return False  # Reject if the face is too small or too large
    
    # Check the aspect ratio of the detected face (width/height ratio)
    aspect_ratio = w / h
    if aspect_ratio < 0.75 or aspect_ratio > 1.33:  # Assuming normal face aspect ratio
        return False  # Reject if the aspect ratio is abnormal
    
    return True  # Face is properly framed and acceptable

# Main loop to capture video frames and process faces
while True:
    ret, frame = cap.read()  # Read a frame from the video capture
    if not ret:
        break  # Exit loop if frame capture fails
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Check if the entire head is properly framed in the video
        if not is_full_head_in_frame((x, y, w, h), frame):
            print("Head not fully in frame, skipping.")
            continue  # Skip this face if it doesn't meet the criteria
        
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        
        # Check if the face image is blurry
        if is_blurry(face):
            print("Blurry image detected, skipping.")
            continue  # Skip this face if it is blurry
        
        # Resize the face image to the desired resolution (100x100 pixels)
        face_resized = cv2.resize(face, (100, 100))
        
        # Save the processed face image to the prepared directory
        cv2.imwrite(os.path.join(prepared_dir, f"face_{count}.jpg"), face_resized)
        
        count += 1  # Increment the counter for the next image
        break  # Only process the first detected face in each frame

    # Display the current video frame with the detected face (if any)
    cv2.imshow('Face Capture', frame)
    
    # Break the loop and stop capturing if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
