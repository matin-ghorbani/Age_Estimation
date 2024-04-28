from argparse import ArgumentParser, BooleanOptionalAction

import cv2 as cv
from face_detector import FaceDetector


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='The source file or the camera id')
    parser.add_argument('--model', type=str, default='weights/best_age_estimator_resnet.h5', help='The model file path')
    parser.add_argument('--draw', type=bool, action=BooleanOptionalAction, default=True,
                        help='Draw bounding boxes on the image')
    opt = parser.parse_args()

    try:
        opt.source = int(opt.source)
    except ValueError:
        pass

    detector = FaceDetector(opt.model)

    cap = cv.VideoCapture(opt.source)
    while True:
        success, frame = cap.read()
        if success:
            result_frame, bboxs = detector.detect_face(frame, draw=opt.draw)

            if len(bboxs) > 0:
                for bbox in bboxs:
                    x1, y1, x2, y2 = bbox
                    face = frame[y1: y2, x1: x2]
                    age = detector.estimate_age(face)

                    if not opt.draw:
                        org = ((x1 + x2) // 2, (y1 + y2) // 2)
                    else:
                        org = (x1, y2 + 20)

                    cv.putText(result_frame, f'Age: {age}', org,
                            cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)

            cv.imshow('webcam', result_frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

        else:
            # raise Exception('Could not read frames from source file or camera.')
            print('Could not read frames from source file or camera.')
            break

    cap.release()


if __name__ == '__main__':
    main()
