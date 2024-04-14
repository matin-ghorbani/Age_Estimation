import cv2 as cv
from face_detector import FaceDetector

detector = FaceDetector()

cap = cv.VideoCapture(0)
while True:
    frame = cap.read()[1]
    frame, bboxs = detector.detect_face(frame, draw=True)

    cv.imshow('webcam', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
