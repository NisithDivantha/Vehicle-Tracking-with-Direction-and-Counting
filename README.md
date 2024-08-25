# Vehicle-Tracking-and-Counting 🚗
This project involves tracking and counting vehicles in a video using the SORT (Simple Online and Realtime Tracking) algorithm, color histogram features for appearance matching, and OpenCV for video processing and display. In this project I detect the direction of each vehicle and count the number of vehicles that goes in each dierection. This project was undertaken to gain hands-on experience in computer vision.

## Features

- Vehicle detection using pre-recorded detection data.
- Tracking vehicles using the SORT algorithm.
- Matching vehicle appearances using color histograms.
- Counting vehicles moving in different directions (left and right).
- Annotating video frames with bounding boxes, labels, and count information.
- Saving the processed video with annotated information.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- OpenCV
- NumPy
- SORT (Simple Online and Realtime Tracking)
- Collections (defaultdict)
- Pickle

You can install the required Python libraries using pip:

```bash
pip install opencv-python numpy
```

### Files

- detectionHistoryYOLO.pkl: Pre-recorded vehicle detection data.
- Traffic.mp4: Input video file can be found in the Vedio_link.txt file.


Clone the repository:
```bash
git clone https://github.com/NisithDivantha/Vehicle-Tracking-and-Counting.git
```

## Code Explanation
### The main script performs the following steps:

- Imports and Initializations:
- Imports necessary libraries.
- Defines functions to extract color histograms and calculate histogram similarity.
- Initializes the SORT tracker and other variables.

### Main Processing Loop:
- Reads each frame from the video.
- Retrieves and formats detections for the current frame.
- Updates the SORT tracker with the current detections.
- Matches tracked objects based on appearance and updates direction counts.
- Draws bounding boxes, labels, and counts on the frame.

### Final Steps:
- Writes the processed frame to the output video.
- Releases resources and closes all windows.

## Results
The output video (output_with_car_count.mp4) will have bounding boxes around detected vehicles, each labeled with a unique ID. It will also display the total number of unique cars detected and counts of cars moving left and right.

## Motivation
This project was undertaken to gain hands-on experience with computer vision techniques, including object detection, tracking, and appearance matching. The SORT algorithm provided a practical introduction to real-time tracking, while OpenCV facilitated video processing and display.


## Acknowledgments
The SORT algorithm implementation was inspired by the paper "Simple Online and Realtime Tracking" by Bewley et al.
Detection data was simulated for the purpose of this project.
