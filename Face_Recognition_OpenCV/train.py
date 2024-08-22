import cv2
import os
import numpy as np

# Directory where face images are stored
image_dir = r"PATH_TO_FACE_IMAGES_DIRECTORY"
# Directory where the trained model will be saved
output_dir = r"PATH_TO_MODEL_OUTPUT_DIRECTORY"

# Initialize the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to retrieve face images and their corresponding labels
def get_images_and_labels(image_dir):
    face_samples = []
    ids = []
    # Set a fixed ID for all images (can be adjusted for multiple identities)
    person_id = 1
    for image_path in os.listdir(image_dir):
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            img = cv2.imread(os.path.join(image_dir, image_path), cv2.IMREAD_GRAYSCALE)
            # Detect faces in the image using Haar cascade classifier
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(img)
            for (x, y, w, h) in faces:
                # Extract the face region and append it to the list of samples
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(person_id)  # Append the fixed ID for each face
    return face_samples, ids

# Retrieve face images and labels from the specified directory
faces, ids = get_images_and_labels(image_dir)

# Train the face recognizer with the collected faces and their labels
recognizer.train(faces, np.array(ids))

# Save the trained model in the specified output directory
model_path = os.path.join(output_dir, 'face_recognition_model.yml')
recognizer.save(model_path)
print(f"Model trained and saved at {model_path}.")
