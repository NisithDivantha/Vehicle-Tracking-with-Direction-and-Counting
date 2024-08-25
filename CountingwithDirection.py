import cv2
import numpy as np
from sort import Sort  
from collections import defaultdict
import pickle

# Function to extract color histogram features
def extract_color_histogram(image, bbox, bins=(8, 8, 8)):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    hist = cv2.calcHist([roi], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to calculate histogram similarity
def calculate_hist_similarity(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3)

# Set to store unique object IDs and their appearance features
unique_object_ids = set()
appearance_features = defaultdict(dict)

# Load detections from detectionHistoryYOLO.pkl
with open('detectionHistory.pkl', 'rb') as f:
    detection_history = pickle.load(f)

# Load video
cap = cv2.VideoCapture("Traffic.mp4")

# Initialize VideoWriter for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_with_car_count.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

frame_id = 0

# Initialize dictionaries to store previous positions, direction counts, and direction status
prev_positions = {}
direction_counts = {}
direction_status = {}
left_count = 0
right_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_id >= len(detection_history):
        break

    # Get detections for the current frame from detection history
    frame_detections = detection_history[frame_id]

    # Convert detections to the format SORT expects [x1, y1, x2, y2, confidence]
    detections = []
    for detection in frame_detections:
        x, y, w, h = detection
        x1, y1, x2, y2 = x, y, x + w, y + h
        confidence = 1.0  
        detections.append([x1, y1, x2, y2, confidence])
    detections = np.array(detections)
    # Update tracker
    tracked_objects = tracker.update(detections)

    # Match objects based on appearance and update appearance features
    for obj in tracked_objects:
        object_id = int(obj[4])  # SORT assigns object ID as the 5th element
        bbox = obj[:4].astype(int)
        hist = extract_color_histogram(frame, bbox)

        # Check if this object matches any previously tracked object based on appearance
        if object_id not in appearance_features:
            # If it's a new object, add it to the unique IDs set and store its appearance feature
            unique_object_ids.add(object_id)
            appearance_features[object_id]['hist'] = hist
            # Initialize direction status for the new object
            direction_status[object_id] = None
        else:
            # If it's a known object, update its appearance feature
            similarity = calculate_hist_similarity(appearance_features[object_id]['hist'], hist)
            if similarity < 0.7: 
                object_id = max(unique_object_ids) + 1
                unique_object_ids.add(object_id)
                appearance_features[object_id]['hist'] = hist
                direction_status[object_id] = None  # Reset direction status for new ID
            else:
                appearance_features[object_id]['hist'] = hist

        # Determine direction and update counts if not previously counted
        current_position = (bbox[0] + bbox[2]) // 2  # x-center of the bounding box
        if object_id in prev_positions:
            prev_position = prev_positions[object_id]
            if direction_status.get(object_id) is None:
                if current_position > prev_position:
                    direction_status[object_id] = "right"
                    right_count += 1
                elif current_position < prev_position:
                    direction_status[object_id] = "left"
                    left_count += 1
        # Update previous position
        prev_positions[object_id] = current_position

    total_cars_detected = len(unique_object_ids)

    # Draw bounding boxes, labels, and car count
    for obj in tracked_objects:
        x1, y1, x2, y2, object_id = obj.astype(int)
        label = f"ID {int(object_id)}"
        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Create a background rectangle for the total cars count
    cv2.rectangle(frame, (800, 800), (1050, 880), (255, 255, 255), -1)
    cv2.putText(frame, f"Total cars: {total_cars_detected}", (820, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Create a background rectangle for the left count
    cv2.rectangle(frame, (500, 800), (750, 880), (255, 255, 255), -1)
    cv2.putText(frame, f"Left: {left_count}", (570, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Create a background rectangle for the right count
    cv2.rectangle(frame, (1100, 800), (1350, 880), (255, 255, 255), -1)
    cv2.putText(frame, f"Right: {right_count}", (1160, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
