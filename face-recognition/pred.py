"""
pred.py
"""

import collections

import numpy as np
from scipy.ndimage import rotate
import cv2

from helper.detection import get_rotation_map, detect_faces, trim
from helper.data import name_svm

NAMES = collections.Counter()
MOST = 5
TITLE = "Face Recognition"
SKIP_FRAME = 2  # the fixed skip frame
SCALE_FACTOR = 2  # used to resize the captured frame
KEYS = [27, ord('q'), ord('Q')]
INTERPOLATION = cv2.INTER_AREA
COLOR_GRAY = cv2.COLOR_BGR2GRAY
FONT = cv2.FONT_HERSHEY_SIMPLEX


def live_svm():
    counter = 0
    skip_rate = 0  # skip SKIP_FRAME frames every other frame
    rotation_map = get_rotation_map()
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()  # get first frame
    scale = (frame.shape[1] // SCALE_FACTOR,
             frame.shape[0] // SCALE_FACTOR)  # (y, x)
    while ret:
        key = cv2.waitKey(100)
        if key in KEYS:
            break
        resized_frame = cv2.resize(frame, scale, interpolation=INTERPOLATION)
        resized_frame = cv2.flip(resized_frame, 1)
        processed_frame = resized_frame
        if not skip_rate:
            facefound = False
            for r in rotation_map:
                rotated_frame = rotate(resized_frame, r)
                gray_frame = cv2.cvtColor(rotated_frame, COLOR_GRAY)
                faces = detect_faces(gray_frame)
                if list(faces):
                    one = (len(faces) == 1)
                    for f in faces:
                        x, y, w, h = [v for v in f]
                        y_i, x_i = (y + h, x + w)
                        cropped_face = gray_frame[y:y_i, x:x_i]
                        cropped_face = cv2.flip(cropped_face, 1)
                        name_predict = name_svm(cropped_face)
                        if one:
                            new_name = {name_predict: 1}
                            NAMES.update(new_name)
                        cv2.rectangle(rotated_frame, (x, y), (x_i, y_i),
                                      (0, 255, 0))
                        if counter > SKIP_FRAME:
                            if one:
                                name_values = [
                                    count
                                    for _, count in NAMES.most_common(MOST)
                                ]
                                step = len(name_values) - 1
                                if step == 0:
                                    step = 1
                                avg = np.arange(99.9, 0.9, -98.9 / step)
                                avg = np.resize(avg, np.shape(name_values)[0])
                                name_percent = int(
                                    np.average(avg, weights=name_values))
                                common_names = NAMES.most_common(1)[0][0]
                                name_to_display = str(
                                    name_percent) + "% " + common_names
                            else:
                                name_to_display = name_predict
                            cv2.putText(rotated_frame, name_to_display, (x, y),
                                        FONT, 0.7, (0, 255, 0))
                    processed_frame = rotate(
                        rotated_frame, (-r), reshape=False)
                    rotation_map = get_rotation_map(r)
                    facefound = True
            if facefound:
                skip_rate = 0
                counter += 1
            else:
                skip_rate = SKIP_FRAME
                counter = 0
                NAMES.clear()
        else:
            skip_rate -= 1
        processed_frame = trim(processed_frame, scale)
        cv2.putText(processed_frame, "Press ESC or 'q' to quit.", (5, 15),
                    FONT, 0.4, (255, 255, 255))
        cv2.imshow(TITLE, processed_frame)
        ret, frame = cam.read()  # get next frame
    cam.release()
    cv2.destroyAllWindows()
