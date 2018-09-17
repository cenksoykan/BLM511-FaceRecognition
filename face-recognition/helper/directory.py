"""
helper/directory.py
"""

from sys import stderr, stdout
from errno import EEXIST, ENOTEMPTY

EXTENSIONS = [".png", ".jpg", ".jpeg", ".pgm"]
IMG_MIN = 10
PROFILE_MIN = 2


def make_directory(directory):
    try:
        directory.mkdir()
    except FileExistsError as exception:
        if exception.errno is not EEXIST:
            raise OSError(f"Could not make {directory.name!r} directory")
        print(
            f"\n* INFO: {directory.name!r} already exists, it will be added to\n",
            file=stdout)
    else:
        print(
            f"\n* INFO: A new directory {directory.name!r} is created\n",
            file=stdout)


def remove_directory(directory):
    try:
        directory.rmdir()
    except OSError as exception:
        if exception.errno is not ENOTEMPTY:
            raise OSError(f"Could not remove {directory.name!r} directory")
        print(
            f"\n* ERROR: {directory.name!r} is not empty",
            "but also it does not contain any valid image\n",
            file=stderr)
    else:
        print(
            f"\n* WARNING: {directory.name!r} is deleted",
            "because it does not contain any valid image\n",
            file=stdout)


def clean_directory(directory):
    profiles = list(directory.iterdir())
    for profile in profiles:
        if profile.is_file():
            profile.unlink()
        elif profile.is_dir():
            remove_directory(profile)
