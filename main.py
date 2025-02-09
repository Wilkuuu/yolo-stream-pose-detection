# import cv2
# import torch
# from ultralytics import YOLO
# # import torch
# import utils
#
#
#
# model = YOLO("yolo11l-pose.pt")
#
# # Adjust brightness (beta) and contrast (alpha)
# def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
#     """
#     Adjust the brightness and contrast of the frame.
#     alpha: Contrast control (1.0-3.0)
#     beta: Brightness control (0-100)
#     """
#     return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
#
# KEYPOINT_LABELS = [
#     "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
#     "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
#     "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
#     "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
# ]
#
#
# def main():
#     droidcam_url = "http://192.168.1.20:4747/video"
#     cap = cv2.VideoCapture(droidcam_url)
#     # cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default webcam)
#     alpha = .7  # Contrast
#     beta = .2
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
#
#         frame = adjust_brightness_contrast(frame, alpha=alpha, beta=beta)
#         # Run pose detection
#         results = model(frame)  # Inference with YOLOv8 pose model
#         results_tensor = torch.tensor(results[0].keypoints.xy)
#         kp = results_tensor[0]  # Shape: (17, 2)
#
#         # Merge keypoints with labels
#         merged_keypoints = {label: tuple(coord.tolist()) for label, coord in zip(KEYPOINT_LABELS, kp)}
#
#         # Display the result
#         for label, coord in merged_keypoints.items():
#             print(f"{label}: {coord}")
#
#         annotated_frame = results[0].plot()  # YOLOv8 provides a plotting function to visualize results
#
#         # Display the resulting frame
#         cv2.imshow('Pose Detection', annotated_frame)
#
#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     # Release the capture and destroy all windows
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

# Load the YOLO pose model
model = YOLO("yolo11l-pose.pt")

KEYPOINT_LABELS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Buffer to store past positions
POSE_HISTORY = deque(maxlen=30)  # Store last 3 seconds (assuming ~10 FPS)

# Keypoints to check for fall detection
CHECK_KEYPOINT_LABELS = [
     "Nose", "Left Eye"
]


def detect_fall(keypoints):
    """
    Detects falls by tracking multiple keypoints.
    Checks for sudden downward movement of the body and loss of upright posture.
    Handles cases where some keypoints may be missing.
    """
    if len(POSE_HISTORY) < 5:
        return False  # Not enough data for detection

    # Collect keypoints safely (skip missing ones)
    current_positions = {
        label: np.array(keypoints[label])
        for label in CHECK_KEYPOINT_LABELS if label in keypoints
    }

    if len(current_positions) < 5:  # Ensure we have enough keypoints
        return False

    # Get previous positions, skipping frames with missing keypoints
    prev_positions_list = []
    for p in POSE_HISTORY:
        prev_positions = {
            label: np.array(p[label])
            for label in CHECK_KEYPOINT_LABELS if label in p
        }
        if len(prev_positions) >= 5:
            prev_positions_list.append(prev_positions)

    if len(prev_positions_list) < 5:  # Ensure enough valid frames
        return False

    # Compute displacement and velocity for each tracked keypoint
    total_displacement = 0
    total_velocity = 0
    valid_keypoints = 0

    for label, curr_pos in current_positions.items():
        prev_pos = np.array([prev[label] for prev in prev_positions_list if label in prev])

        if len(prev_pos) > 1:
            displacement = prev_pos[-1][1] - curr_pos[1]  # Y-axis drop
            velocity = np.mean(np.diff(prev_pos[:, 1]))  # Speed of fall

            total_displacement += displacement
            total_velocity += velocity
            valid_keypoints += 1

    if valid_keypoints == 0:
        return False

    avg_displacement = total_displacement / valid_keypoints
    avg_velocity = total_velocity / valid_keypoints

    # Fall detection thresholds
    FALL_THRESHOLD = 30  # Min Y-axis displacement (falling down)
    VELOCITY_THRESHOLD = 10  # Speed threshold

    # Check if multiple keypoints moved downward quickly
    print(avg_displacement , FALL_THRESHOLD, avg_velocity , VELOCITY_THRESHOLD)
    time.sleep(1)
    if avg_displacement > FALL_THRESHOLD and avg_velocity > VELOCITY_THRESHOLD:
        return True

    return False


def main():
    droidcam_url = "http://192.168.1.20:4747/video"
    cap = cv2.VideoCapture(droidcam_url)

    alpha = 0.7  # Contrast
    beta = 0.2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        results = model(frame)

        if len(results) > 0 and results[0].keypoints is not None:
            kp_tensor = torch.tensor(results[0].keypoints.xy)
            kp = kp_tensor[0]  # (17, 2) shape

            # Convert to dictionary
            keypoints = {label: tuple(coord.tolist()) for label, coord in zip(KEYPOINT_LABELS, kp)}

            # Store keypoints in history
            POSE_HISTORY.append(keypoints)

            # Check for fall detection
            if detect_fall(keypoints):
                print("⚠️ Fall Detected! Pausing for 3 seconds...")
                time.sleep(3)  # Pause execution for 3 seconds

            # Draw keypoints
            annotated_frame = results[0].plot()
            cv2.imshow('Pose Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
