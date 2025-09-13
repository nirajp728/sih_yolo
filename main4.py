from ultralytics import YOLO 
import cvzone
import cv2
import numpy as np
from collections import deque

# Load YOLO model
model = YOLO('yolov10n.pt')

# Webcam capture
cap = cv2.VideoCapture(0)

# Create a named window that stays on top
cv2.namedWindow("Recording - Press Q to Quit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Recording - Press Q to Quit", cv2.WND_PROP_TOPMOST, 1)

prev_gray = None
motion_history = deque(maxlen=5)  # store last 5 motion values

while True:
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    n = 1  # assume stable
    avg_motion = 0  # default in case no motion is calculated yet

    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        diff_mean = np.mean(frame_diff)

        motion_history.append(diff_mean)
        avg_motion = np.mean(motion_history)

        if avg_motion > 6:  
            n = 0  # shaky
        else:
            n = 1  # stable

    prev_gray = gray


    # Run detection
    results = model(image, verbose=False)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', 
                               [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Show recording window with n value
    cv2.putText(image, f"n = {n}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if n==1 else (0, 0, 255), 2)
    cv2.imshow("Recording - Press Q to Quit", image)

    print(f"n = {n}, avg_motion = {avg_motion:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
