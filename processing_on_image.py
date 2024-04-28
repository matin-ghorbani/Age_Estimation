import datetime
import os
from argparse import ArgumentParser, BooleanOptionalAction

import cv2 as cv
from face_detector import FaceDetector


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--img', type=str, default='io/inputs/MESSI.jpg', help='Image to process')
    parser.add_argument('--model', type=str, default='weights/best_age_estimator_resnet.h5', help='The model file path')
    parser.add_argument('--draw', type=bool, action=BooleanOptionalAction, default=True,
                        help='Draw bounding boxes on the image')
    parser.add_argument('--save', type=bool, action=BooleanOptionalAction, default=True,
                        help='Save the processed image')
    parser.add_argument('--show', type=bool, action=BooleanOptionalAction, default=True,
                        help='Show the processed image')
    opt = parser.parse_args()

    detector = FaceDetector(opt.model)
    img: cv.typing.MatLike = cv.imread(opt.img)
    result_img, bboxs = detector.detect_face(img, draw=opt.draw)

    if len(bboxs) > 0:
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            face = img[y1: y2, x1: x2]
            age = detector.estimate_age(face)

            if not opt.draw:
                org = ((x1 + x2) // 2, (y1 + y2) // 2)
            else:
                org = (x1, y2 + 20)

            cv.putText(result_img, f'Age: {age}', org,
                       cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)

    if opt.show:
        cv.imshow('Result Image', result_img)
        cv.waitKey(0)

    if opt.save:
        time = datetime.datetime.now().strftime('%H_%M_%S')
        output_path = os.path.join('.', 'io', 'outputs', f'result_img_{time}.jpg')
        is_saved = cv.imwrite(output_path, result_img)
        if is_saved:
            print(f'Result image saved to {output_path}')
        else:
            print(f'Could not save the result image to {output_path}')


if __name__ == '__main__':
    main()
