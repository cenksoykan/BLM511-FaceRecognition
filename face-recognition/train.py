"""
train.py
"""

from scipy.ndimage import rotate
import cv2

from helper.detection import get_rotation_map, detect_faces, trim
from helper.image import check_format
from helper.profile import create_profile

DISPLAY_FACE_DIM = (128, 128)
TITLE = "Face Recognition"
SKIP_FRAME = 2  # the fixed skip frame
SCALE_FACTOR = 2  # used to resize the captured frame
KEYS = [27, ord('q'), ord('Q')]
INTERPOLATION = cv2.INTER_AREA
COLOR_GRAY = cv2.COLOR_BGR2GRAY
FONT = cv2.FONT_HERSHEY_SIMPLEX


def add_profile(profile_name):
    profile = create_profile(profile_name)
    saved_face = 0
    pictures = list(profile.iterdir())
    for picture in pictures:
        if check_format(picture):
            saved_face += 1
    unsaved = True
    skip_rate = 0  # skip SKIP_FRAME frames every other frame
    rotation_map = get_rotation_map()
    cropped_face = []
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
                    for f in faces:
                        x, y, w, h = [v for v in f]
                        y_i, x_i = (y + h, x + w)
                        cropped_face = gray_frame[y:y_i, x:x_i]
                        cropped_face = cv2.resize(
                            cropped_face,
                            DISPLAY_FACE_DIM,
                            interpolation=INTERPOLATION)
                        cropped_face = cv2.flip(cropped_face, 1)
                        cv2.rectangle(rotated_frame, (x, y), (x_i, y_i),
                                      (0, 255, 0))
                        cv2.putText(rotated_frame, "Training Face", (x, y),
                                    FONT, 0.7, (0, 255, 0))
                    processed_frame = rotate(rotated_frame, (-r))
                    rotation_map = get_rotation_map(r)
                    facefound = True
            if facefound:
                skip_rate = 0
                unsaved = True
            else:
                skip_rate = SKIP_FRAME
        else:
            skip_rate -= 1
        processed_frame = trim(processed_frame, scale)
        cv2.putText(processed_frame, "Press ESC or 'q' to quit.", (5, 15),
                    FONT, 0.4, (255, 255, 255))
        cv2.imshow(TITLE, processed_frame)
        if list(cropped_face):
            cv2.imshow("Recognized Face", cropped_face)
            if unsaved and key in [ord('p'), ord('P')]:
                face_name = profile_name + "-" + str(saved_face) + ".pgm"
                img_path = str(profile.joinpath(face_name))
                cv2.imwrite(img_path, cropped_face)
                saved_face += 1
                unsaved = False
                print("Saved:", face_name)
        ret, frame = cam.read()  # get next frame
    cam.release()
    cv2.destroyAllWindows()
