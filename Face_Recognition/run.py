import cv2
import time

# Load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('PATH_TO_face_recognition_model.yml')  # Load the pre-trained model for face recognition

# Initialize the face detector using Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video source (0 = default camera, replace with your IP stream if needed)
vid = 0

# Start video capture from the specified video source
cap = cv2.VideoCapture(vid)

# Timer for FPS calculation
prev_frame_time = 0  # Time of the previous frame
new_frame_time = 0  # Time of the current frame

# Main loop for video capture and face recognition
while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the video stream
    if not ret:
        break  # Exit the loop if frame capture fails

    # Update the time for the new frame to calculate FPS
    new_frame_time = time.time()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using the Haar cascade
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale image
        face = gray[y:y+h, x:x+w]
        
        # Resize the face region to the required size for the recognizer (100x100 pixels)
        face_resized = cv2.resize(face, (100, 100))
        
        # Use the recognizer to predict the ID and confidence of the detected face
        id, confidence = recognizer.predict(face_resized)
        
        # Determine the name and confidence level based on the recognizer's prediction
        if confidence < 70:  # If the confidence is good (lower value means better match)
            name = "Unknown"  # Label as "Unknown" if the match is weak
            confidence_text = f"{round(100 - confidence)}%"
        elif confidence < 100:  # Acceptable confidence range
            name = f"ID {id}"  # Display the recognized ID
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"  # Label as "Unknown" if the confidence is too low
            confidence_text = f"{round(100 - confidence)}%"

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the name and confidence level on the frame
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
        
        # Print the ID and confidence to the console for debugging
        print(f"ID: {id}, Confidence: {confidence}")

    # Calculate the Frames Per Second (FPS)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display the calculated FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with face recognition in a window
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop and stop video capture if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
