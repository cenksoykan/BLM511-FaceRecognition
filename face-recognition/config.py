"""
config.py
"""

from pathlib import Path

ROOT_PATH = Path(__file__).parent.joinpath("..")
DATABASE_PATH = ROOT_PATH.joinpath("db")
CACHE_FOLDER = DATABASE_PATH.joinpath("cache")
DATA_FOLDER = DATABASE_PATH.joinpath("src")
OMIT_FOLDER = "."
OMIT_FILE = "Ambient"
EXTENSIONS = [".png", ".jpg", ".jpeg", ".pgm"]

VERBOSE: bool = True
K: int = 80
TEST_SIZE: float = 0.2
FACE_DIM: tuple = (50, 50)
THRESHOLD: int = 5000


class Svm:
    KERNEL = "rbf"  # linear, poly, rbf, sigmoid, precomputed
    C = [1e3, 5e3, 1e4, 5e4, 1e5]  # Penalty parameter C of the error term
    # Kernel coefficient for rbf, poly and sigmoid
    # If gamma is auto then 1/n_features will be used instead.
    GAMMA = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.1]


class Face:
    PROFILE = "yaleB05"
    NAME = PROFILE + "_P00A-005E-10"
    SUFFIX = ".pgm"
    IMG = NAME + SUFFIX


SVM = Svm
FACE = Face
