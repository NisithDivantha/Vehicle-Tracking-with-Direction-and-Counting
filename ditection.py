import cv2
import numpy as np
import pickle

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
cap = cv2.VideoCapture("Traffic.mp4")
detection_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocess the frame (e.g., increase contrast)
    # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Populate the lists with detection data
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] == "car":  # Lower threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Max Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    frame_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            frame_detections.append(box)

    # Save detections to history, or if none, interpolate from previous frames (advanced step)
    if len(frame_detections) == 0:
        # Apply interpolation or tracking methods here if needed
        pass
    
    detection_history.append(frame_detections)

    # Optionally draw boxes on frame (for visualization)
    for i, box in enumerate(frame_detections):
        x, y, w, h = box
        label = "car"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the detection history to a file
with open('detectionHistory.pkl', 'wb') as f:
    pickle.dump(detection_history, f)

print("Detection history saved to detectionHistoryYOLO.pkl")
