import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# Load the YOLO model
yolo_model = YOLO("yolov8n.pt")

previous_vehicle_centers = {}
vehicle_speeds = defaultdict(lambda: 0)
collision_alerts = {}
frame_times = {}
fixed_distance_pixels = 50  # Distance in pixels for speed calculation
real_distance_meters = 5.0  # Real-world distance in meters for speed calculation
collision_displayed = False  # Track if a collision warning has been displayed

# Color definitions for drawing
COLOR_CORAL = (80, 127, 255)
COLOR_CHARCOAL = (79, 69, 54)
COLOR_RED = (0, 0, 255)

# List of vehicle classes to be detected
detected_vehicle_classes = ["car", "motorbike", "bus", "truck", "bicycle", "scooter", "auto", "person"]

def detect_objects_in_frame(frame):
    results = yolo_model(frame)
    return results

def extract_bounding_boxes(results):
    class_ids = []
    confidences = []
    bounding_boxes = []
    for result in results:
        for detection in result.boxes:
            x, y, w, h = detection.xywh.numpy().astype(int).flatten()
            confidence = detection.conf.item()
            class_id = detection.cls.item()
            if yolo_model.names[class_id] in detected_vehicle_classes:
                class_ids.append(class_id)
                confidences.append(confidence)
                bounding_boxes.append([x - w // 2, y - h // 2, w, h])
    return class_ids, confidences, bounding_boxes

def draw_boxes_and_annotations(frame, class_ids, confidences, bounding_boxes, avg_vehicle_width):
    global frame_times, collision_displayed
    font_small = cv2.FONT_HERSHEY_PLAIN
    font_large = cv2.FONT_HERSHEY_DUPLEX
    current_vehicle_centers = {}
    current_time = time.time()

    for i in range(len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i]
        label = str(yolo_model.names[int(class_ids[i])])
        color = COLOR_CORAL
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), font_small, 1.5, COLOR_CHARCOAL, 2)
        current_vehicle_centers[i] = (x + w / 2, y + h / 2)
        
        # Calculate the distance to the vehicle
        distance = (avg_vehicle_width * 720) / w  # Assuming 720 as focal length
        cv2.putText(frame, f"{distance:.2f}m", (x, y + h + 20), font_small, 1.5, COLOR_CHARCOAL, 2)

        # Speed estimation using the time taken to move a fixed distance
        if i in previous_vehicle_centers:
            prev_center = previous_vehicle_centers[i]
            curr_center = current_vehicle_centers[i]
            delta_x = abs(curr_center[0] - prev_center[0])
            
            if delta_x > fixed_distance_pixels:
                time_diff = current_time - frame_times[i]
                speed = (real_distance_meters / time_diff) * 3.6  # Convert to km/h
                if 30 <= speed <= 70:  
                    vehicle_speeds[i] = speed
                    cv2.putText(frame, f"{speed:.2f}km/h", (x, y + h + 40), font_small, 1.5, COLOR_CHARCOAL, 2)
                    
                    # Calculate Time to Collision (TTC)
                    ttc = distance / (speed / 3.6)  # Convert speed from km/h to m/s
                    if 0.5 <= ttc <= 0.7:  # Display alert for the required TTC range
                        collision_alerts[i] = current_time
                        collision_displayed = True  # Display only one alert at a time

    # Display collision warning if applicable
    for i in list(collision_alerts.keys()):
        if current_time - collision_alerts[i] <= 5 and collision_displayed:  
            if i < len(bounding_boxes):  
                x, y, w, h = bounding_boxes[i]
                if distance < 10:  # Check if the vehicle is very close
                    warning_text = "Collision Alert!"
                    text_size = cv2.getTextSize(warning_text, font_large, 1.5, 2)[0]
                    text_x = x
                    text_y = y + h + 60
                    cv2.rectangle(frame, (text_x, text_y - text_size[1]), 
                                  (text_x + text_size[0], text_y + text_size[1] // 2), 
                                  (0, 0, 0), -1)
                    cv2.putText(frame, warning_text, (text_x, text_y), font_large, 1.5, COLOR_RED, 2)
        else:
            del collision_alerts[i]
            collision_displayed = False  # Reset the flag once the alert time has passed

    previous_vehicle_centers.update(current_vehicle_centers)
    frame_times.update({i: current_time for i in current_vehicle_centers})
    return frame, current_vehicle_centers

# Process the video file
def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    global previous_vehicle_centers

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    avg_vehicle_width = 2.0  # Average vehicle width in meters

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects_in_frame(frame)
        class_ids, confidences, bounding_boxes = extract_bounding_boxes(results)

        frame, previous_vehicle_centers = draw_boxes_and_annotations(
            frame, class_ids, confidences, bounding_boxes, avg_vehicle_width)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the image folder
def process_image_folder(image_folder):
    global previous_vehicle_centers

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        results = detect_objects_in_frame(frame)
        class_ids, confidences, bounding_boxes = extract_bounding_boxes(results)
        frame, previous_vehicle_centers = draw_boxes_and_annotations(frame, class_ids, confidences, bounding_boxes, avg_vehicle_width=2.0)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Check if the input contains image dataset or video dataset
def main(input_path):
    if os.path.isdir(input_path):
        process_image_folder(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video_file(input_path)
    else:
        print("Invalid input. Please provide a path to an image dataset folder or a video file.")

input_path = "input_path_for_testing"  # Replace with input path
main(input_path)


