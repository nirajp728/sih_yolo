from ultralytics import YOLO 
import cvzone
import cv2
import numpy as np

# Load YOLO model
model = YOLO('yolov10n.pt')

# Webcam capture
cap = cv2.VideoCapture(0)

# Create a named window that stays on top
cv2.namedWindow("Recording - Press Q to Quit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Recording - Press Q to Quit", cv2.WND_PROP_TOPMOST, 1)

prev_gray = None  # to store previous frame for motion detection

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Convert to grayscale for motion detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    n = 1  # default: stable
    if prev_gray is not None:
        # Absolute difference between current frame and previous frame
        frame_diff = cv2.absdiff(prev_gray, gray)
        diff_mean = np.mean(frame_diff)

        # Threshold: adjust based on how sensitive you want it
        if diff_mean > 10:   # 10 is arbitrary, tune for your case
            n = 0  # shaky
        else:
            n = 1  # stable

    prev_gray = gray  # update for next iteration

    # Run detection
    results = model(image, verbose=False)  # suppress YOLO logs
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = int(box.conf[0].numpy() * 100)
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', 
                               [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Show recording window
    cv2.putText(image, f"n = {n}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if n==1 else (0, 0, 255), 2)
    cv2.imshow("Recording - Press Q to Quit", image)

    # Print log with n value
    print(f"n = {n}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
