"""
helper/image.py
"""

from sys import stderr
import warnings

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize

from config import FACE_DIM, OMIT_FILE, EXTENSIONS


def check_format(file) -> bool:
    """Check if image format is one of these: png, jpg, jpeg, pgm"""
    extension = file.suffix
    return (extension in EXTENSIONS) and (OMIT_FILE not in file.stem)


def check_path(data_path) -> list:
    if not data_path.exists():
        print(
            "\n* ERROR: There is no picture in this direction\n", file=stderr)
        exit()
    if not check_format(data_path):
        print(
            "\n* ERROR: File extension has to be one of these:\n",
            *EXTENSIONS,
            file=stderr)
        exit()
    return imread(data_path, as_gray=True)


def fetch_image(face: list):
    # # # TODO: remove anti-aliasing on the next version scikit-image@0.15.0 # # #
    image = resize(face, FACE_DIM, mode='reflect', anti_aliasing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = img_as_ubyte(image)
    return image.ravel()
