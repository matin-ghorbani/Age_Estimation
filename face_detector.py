import cv2 as cv


class FaceDetector:
    def __init__(self) -> None:
        self.face_detector = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )

    def detect_face(self, img: cv.typing.MatLike, draw: bool = True) -> list[
        cv.typing.MatLike, list[list[int, int, int, int]]]:
        bboxs: list[list[int, int, int, int]] = []

        result_img = img.copy()

        frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(frame_gray, 1.3)
        for face in faces:
            x, y, w, h = face
            x, w = x - 20, w + 20
            y, h = y - 20, h + 40

            bboxs.append((x, y, w + x, h + y))
            if draw:
                result_img = FaceDetector.draw(img.copy(), (x, y, w + x, h + y))

        return result_img, bboxs

    @staticmethod
    def draw(img, bbox) -> cv.typing.MatLike:
        x1, y1, x2, y2 = bbox
        img = cv.rectangle(
            img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return img


if __name__ == '__main__':
    detector = FaceDetector()

    cap = cv.VideoCapture(0)
    while True:
        frame = cap.read()[1]
        frame, bboxs = detector.detect_face(frame, draw=True)

        print(f'{bboxs = }')

        cv.imshow('webcam', frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
