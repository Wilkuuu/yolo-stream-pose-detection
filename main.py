from time import sleep

import cv2
import torch
from ultralytics import YOLO
# import torch
import utils
import os

image_dir = 'images'

model = YOLO("yolo11l-pose.pt")


# Adjust brightness (beta) and contrast (alpha)
def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    """
    Adjust the brightness and contrast of the frame.
    alpha: Contrast control (1.0-3.0)
    beta: Brightness control (0-100)
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


KEYPOINT_LABELS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


def main():
    # droidcam_url = "http://192.168.8.120:4747/video"
    # cap = cv2.VideoCapture(droidcam_url)
    # cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default webcam)
    # alpha = .7  # Contrast
    # beta = .2
    for f in os.listdir(image_dir):
        image_path = os.path.join(image_dir, f)
        frame = cv2.imread(image_path)
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("Failed to grab frame")
        #         break

        # frame = adjust_brightness_contrast(frame, alpha=alpha, beta=beta)
        # Run pose detection
        results = model(frame)  # Inference with YOLOv8 pose model
        results_tensor = torch.tensor(results[0].keypoints.xy)
        kp = results_tensor[0]  # Shape: (17, 2)

        # Merge keypoints with labels
        merged_keypoints = {label: tuple(coord.tolist()) for label, coord in zip(KEYPOINT_LABELS, kp)}

        # Display the result
        for label, coord in merged_keypoints.items():
          print(f"{label}: {coord}")

        annotated_frame = results[0].plot()  # YOLOv8 provides a plotting function to visualize results

        # Display the resulting frame
        cv2.imshow('Pose Detection', annotated_frame)

        # Break the loop on 'q' key press
        sleep(1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
