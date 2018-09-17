"""
helper/detection.py
"""

from functools import lru_cache

import cv2

FRONTALFACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILEFACE = cv2.data.haarcascades + "haarcascade_profileface.xml"
FACE_CASCADE = cv2.CascadeClassifier(FRONTALFACE)
SIDEFACE_CASCADE = cv2.CascadeClassifier(PROFILEFACE)
FLAGS = cv2.CASCADE_SCALE_IMAGE
ROTATION_MAPS = {
    "middle": [-30, 30, 0],
    "left": [30, 0, -30],
    "right": [-30, 0, 30],
}


def trim(image: str, dim: tuple):
    """Trim the four sides(black paddings) of the image matrix
    and crop out the middle with a new dimension

    Parameters
    ----------
    image: string
        the image rgb matrix

    dim: tuple (int, int)
        The new dimen the image is trimmed to
    """
    x, y = image.shape[:2]
    y_i, x_i = dim
    y, x = max(0, y - y_i) // 2, max(0, x - x_i) // 2
    return image[x:x + x_i, y:y + y_i]


@lru_cache(maxsize=256)
def get_rotation_map(rotation: int = 0):
    """Takes in an angle rotation, and returns an optimized rotation map"""
    if rotation > 0:
        return ROTATION_MAPS.get("right")
    if rotation < 0:
        return ROTATION_MAPS.get("left")
    return ROTATION_MAPS.get("middle")


def detect_faces(frame):
    faces = FACE_CASCADE.detectMultiScale(
        frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=FLAGS)
    if not list(faces):
        faces = SIDEFACE_CASCADE.detectMultiScale(
            frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=FLAGS)
    return faces
