import cv2
from ultralytics import YOLO
# import torch
import utils



model = YOLO("yolo11l-pose.pt")

# Adjust brightness (beta) and contrast (alpha)
def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    """
    Adjust the brightness and contrast of the frame.
    alpha: Contrast control (1.0-3.0)
    beta: Brightness control (0-100)
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def main():
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default webcam)
    alpha = .7  # Contrast
    beta = .2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = adjust_brightness_contrast(frame, alpha=alpha, beta=beta)
        # Run pose detection
        results = model(frame)  # Inference with YOLOv8 pose model

        # Extract the annotated image from the results
        annotated_frame = results[0].plot()  # YOLOv8 provides a plotting function to visualize results

        # Display the resulting frame
        cv2.imshow('Pose Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()