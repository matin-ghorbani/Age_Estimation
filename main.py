import cv2 as cv
from face_detector import FaceDetector

detector = FaceDetector('./weights/best_age_estimator_resnet.h5')

cap = cv.VideoCapture(0)
while True:
    frame = cap.read()[1]
    result_frame, bboxs = detector.detect_face(frame, draw=True)

    if len(bboxs) > 0:
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            face = frame[y1: y2, x1: x2]
            age = detector.estimate_age(face)
            cv.putText(result_frame, f'Age: {age}', (x1, y2 + 20),
                       cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)

    cv.imshow('webcam', result_frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
