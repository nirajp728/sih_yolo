from ultralytics import YOLO 
import cvzone
import cv2

# Load YOLO model
model = YOLO('yolov10n.pt')

# Webcam capture
cap = cv2.VideoCapture(0)

# Create a named window that stays on top
cv2.namedWindow("Recording - Press Q to Quit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Recording - Press Q to Quit", cv2.WND_PROP_TOPMOST, 1)

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Run detection
    results = model(image)
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
    cv2.imshow("Recording - Press Q to Quit", image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
