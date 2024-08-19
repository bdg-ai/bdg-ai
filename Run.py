import cv2
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import homography as hm 

print(cv2.getBuildInformation())

# Load keypoints from the JSON file (if available)
keypoints_map, keypoints_video = hm.load_keypoints()
dragging_map = -1
dragging_video = -1

# Path to tactical map image and video
tactical_map_path = "2lane.png"
vid_path = r"C:\Data\BDG AI\Projects\Test_BEV\Highway.mp4"

# Load tactical map image
tactical_map = cv2.imread(tactical_map_path)
tactical_map_height, tactical_map_width = tactical_map.shape[:2]

# Load YOLO model
model = YOLO("yolov8x.pt").to('cuda')
names = model.names

# Dictionaries for the last known locations and frame counters of objects
last_known_locations = {}
frame_counters = {}

# Mouse callback functions for moving keypoints
def mouse_callback_map(event, x, y, flags, param):
    global keypoints_map, dragging_map
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if np.linalg.norm([x - keypoints_map[i][0] * tactical_map_width, y - keypoints_map[i][1] * tactical_map_height]) < 10:
                dragging_map = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_map != -1:
            keypoints_map[dragging_map] = [x / tactical_map_width, y / tactical_map_height]
            hm.save_keypoints(keypoints_map, keypoints_video)  # Update JSON file
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_map = -1

def mouse_callback_video(event, x, y, flags, param):
    global keypoints_video, dragging_video, frame_width, frame_height
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if np.linalg.norm([x - keypoints_video[i][0] * frame_width, y - keypoints_video[i][1] * frame_height]) < 10:
                dragging_video = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_video != -1:
            keypoints_video[dragging_video] = [x / frame_width, y / frame_height]
            hm.save_keypoints(keypoints_map, keypoints_video)  # Update JSON file
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_video = -1

# Start video capture
cap = cv2.VideoCapture(vid_path)
assert cap.isOpened(), "Error reading video file"
frame_width, frame_height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Create windows and register mouse callbacks
cv2.namedWindow("Tactical Map")
cv2.setMouseCallback("Tactical Map", mouse_callback_map)

cv2.namedWindow("Video Capture")
cv2.setMouseCallback("Video Capture", mouse_callback_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Draw the square and numbered keypoints on the tactical map
    tactical_map_display = tactical_map.copy()
    normalized_keypoints_map = hm.normalize_points(keypoints_map, tactical_map_width, tactical_map_height)
    cv2.polylines(tactical_map_display, [normalized_keypoints_map], isClosed=True, color=(0, 255, 0), thickness=2)
    for idx, point in enumerate(normalized_keypoints_map):
        cv2.circle(tactical_map_display, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(tactical_map_display, str(idx+1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # YOLO detection
    results = model.track(frame, show=False, tracker="bytetrack.yaml")
    print(results)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(frame, line_width=2, example=names)

    # Update frame_counters for each detected object
    for obj_id in list(last_known_locations.keys()):
        if obj_id not in ids:
            frame_counters[obj_id] += 1
            if frame_counters[obj_id] > 10:
                del last_known_locations[obj_id]
                del frame_counters[obj_id]
        else:
            frame_counters[obj_id] = 0

    if boxes is not None:
        for box, cls, obj_id in zip(boxes, clss, ids):
            if names[int(cls)] == 'truck' or names[int(cls)] == 'car':  	# Only consider trucks and cars
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)]) # Draw bounding box and label

                # Calculate the bottom center point of the bounding box
                bottom_center = hm.get_bottom_center(box)
                # Compute and store the homography matrix
                H, _ = cv2.findHomography(hm.normalize_points(keypoints_video, frame_width, frame_height), normalized_keypoints_map) # Compute homography matrix
                map_point = hm.transform_point(H, bottom_center) # Transform the bottom center point

                # Store the last known location of the object
                last_known_locations[obj_id] = map_point
                frame_counters[obj_id] = 0

                # Draw the original and transformed points
                cv2.circle(frame, (int(bottom_center[0]), int(bottom_center[1])), 5, (255, 0, 0), -1)
    
    # Draw the last known locations on the tactical map
    for map_point in last_known_locations.values():
        cv2.circle(tactical_map_display, (int(map_point[0]), int(map_point[1])), 5, (0, 255, 0), -1)

    # Draw the square and numbered keypoints on the video frame
    normalized_keypoints_video = hm.normalize_points(keypoints_video, frame_width, frame_height)
    cv2.polylines(frame, [normalized_keypoints_video], isClosed=True, color=(0, 255, 0), thickness=2)
    for idx, point in enumerate(normalized_keypoints_video):
        cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(idx+1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the results
    cv2.imshow("Video Capture", frame)
    cv2.imshow("Tactical Map", tactical_map_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
