"""
helper/profile.py
"""

from sys import stderr, stdout

import numpy as np
from skimage.io import imread

from config import DATA_FOLDER, FACE_DIM, OMIT_FOLDER
from .directory import make_directory, remove_directory, clean_directory
from .image import check_format, fetch_image

IMG_MIN = 10
PROFILE_MIN = 2


def read_profile(profile):
    """Reads all the images from one specified face profile"""
    files = list(profile.iterdir())
    for file in files:
        if check_format(file):
            face = imread(file, as_gray=True)
            yield fetch_image(face)


def create_profile(profile_name: str, clean_dir: bool = False) -> str:
    """Create a face profile directory in the database

    Parameters
    ----------
    profile_name: string
        The specified face profile name of a specified face profile folder

    clean_dir: boolean
        Clean the directory if the user already exists

    Returns
    -------
    profile_path: string
        The path of the face profile created
    """
    profile_path = DATA_FOLDER.joinpath(profile_name)
    make_directory(profile_path)
    if clean_dir:
        clean_directory(profile_path)
    return profile_path


def clean_profile(profiles: list) -> list:
    profile_names = []
    for profile in profiles:
        index = 0
        profile_name = profile.name
        files = list(profile.iterdir())
        for file in files:
            if check_format(file):
                index += 1
        if not index:
            remove_directory(profile)
        elif index < IMG_MIN:
            print(
                f"\n* WARNING: Profile {profile_name!r} contains very few images (At least",
                IMG_MIN,
                "images are needed)\n",
                file=stdout)
        else:
            profile_names.append(profile_name)
    return profile_names


def load_profile() -> tuple:
    """Loads all the images from the face profile directory into ndarrays

    Returns
    -------
    x : numpy array, shape = (number_of_faces_in_profiles, pixel_width * pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles

    y : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexes of all the face profile names
    """
    profile_dirs = list(DATA_FOLDER.iterdir())
    profiles = [d for d in profile_dirs if OMIT_FOLDER not in str(d.name)]
    profiles = clean_profile(profiles)
    if len(profiles) < PROFILE_MIN:
        print(
            "\n* ERROR: Database contains very few profiles (At least",
            PROFILE_MIN,
            "profiles are needed)\n",
            file=stderr)
        exit()
    x = np.empty((0, FACE_DIM[0] * FACE_DIM[1]), dtype=np.uint8)
    y = np.empty(0, dtype=np.uint8)
    print("Loading Database:")
    for i, profile_name in enumerate(profiles):
        profile = DATA_FOLDER.joinpath(profile_name)
        x_i = list(read_profile(profile))
        x_i = np.asarray(x_i, dtype=np.uint8)
        size = len(x_i)
        y_i = np.zeros(size, dtype=np.uint8)
        y_i.fill(i)
        x = np.append(x, x_i, axis=0)
        y = np.append(y, y_i)
        print(i + 1, "\t->", size, f"images are loaded from {profile_name!r}")
    return x, y, profiles
