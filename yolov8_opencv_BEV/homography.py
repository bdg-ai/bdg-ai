# homography.py
import cv2
import numpy as np
import json
import os

keypoints_path = "keypoints.json"

# Function to save keypoints to a JSON file
def save_keypoints(keypoints_map, keypoints_video, json_file_path=keypoints_path):
    data = {
        "keypoints_map": keypoints_map.tolist(),
        "keypoints_video": keypoints_video.tolist(),
    }
    with open(json_file_path, 'w') as f:
        json.dump(data, f)

# Function to load keypoints from a JSON file
def load_keypoints(json_file_path=keypoints_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            return np.array(data["keypoints_map"], np.float32), np.array(data["keypoints_video"], np.float32)
    else:
        return np.array([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]], np.float32), np.array([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]], np.float32)

# Function to normalize keypoints
def normalize_points(points, width, height):
    return np.array([[int(x * width), int(y * height)] for x, y in points], np.int32)

# Function to denormalize keypoints
def denormalize_points(points, width, height):
    return np.array([[x / width, y / height] for x, y in points], np.float32)

# Function to transform coordinates using a homography matrix
def transform_point(H, point):
    point_homogeneous = np.array([point[0], point[1], 1.0])
    transformed_point_homogeneous = np.dot(H, point_homogeneous)
    transformed_point = transformed_point_homogeneous[:2] / transformed_point_homogeneous[2]
    return transformed_point

# Function to calculate the bottom center point of the bounding box
def get_bottom_center(box):
    x_center = box[2]
    y_bottom = box[3]
    return (x_center, y_bottom)
