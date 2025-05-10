import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

yolo_model=YOLO('yolov8n.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.5)

# Hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

class ObjectFaceHandDetector:
    def __init__(self):
        self.yolo_model = yolo_model
        self.face_cascade = face_cascade
        self.hands = hands

    def detect_objects(self, frame):
        return self.yolo_model(frame)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5)

    def detect_hands(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(image_rgb)

    def process_frame(self, frame):
        # YOLO object detection
        results = self.detect_objects(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, cls_id in zip(boxes, class_ids):
                label = result.names[cls_id]
                if label != 'person':  # ✅ Only exclude person
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    width = x2 - x1
                    distance = self.estimate_distance(width)
                    cv2.putText(frame, f'Distance: {distance:.2f} cm', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Face detection
        faces = self.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            distance = self.estimate_distance(w)
            cv2.putText(frame, f'Distance: {distance:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hand detection
        hand_results = self.detect_hands(frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    x1, y1 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
                    x2, y2 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                wrist = hand_landmarks.landmark[0]
                tip = hand_landmarks.landmark[12]
                wrist_x = int(wrist.x * frame.shape[1])
                wrist_y = int(wrist.y * frame.shape[0])
                tip_x = int(tip.x * frame.shape[1])
                tip_y = int(tip.y * frame.shape[0])

                distance = self.estimate_distance_hand(wrist_x, wrist_y, tip_x, tip_y)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', (wrist_x, wrist_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def estimate_distance(self, width, focal_length=700):
        real_width = 50  # Example average object width in cm
        return (real_width * focal_length) / width

    def estimate_distance_hand(self, x1, y1, x2, y2, focal_length=700):
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_width = 20  # Hand width assumption
        return (real_width * focal_length) / pixel_distance

def run_detector_on_webcam():
    cap = cv2.VideoCapture(0)
    detector = ObjectFaceHandDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detector.process_frame(frame)
        cv2.imshow('Detection (Objects except Person, Face, Hand)', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run it
run_detector_on_webcam()
