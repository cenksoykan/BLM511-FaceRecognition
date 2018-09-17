"""
helper/data.py
"""

from pickle import dump, load
from sys import stdout

import numpy as np
from sklearn.model_selection import train_test_split
from skimage import img_as_float

from config import CACHE_FOLDER, K, TEST_SIZE
from build import build_pca, build_svm, pred_pca
from .profile import load_profile
from .image import fetch_image


def data_path(file_name: str):
    return CACHE_FOLDER.joinpath(file_name + ".pkl")


def make_file(file_name: str, profile: tuple):
    if not CACHE_FOLDER.exists():
        CACHE_FOLDER.mkdir()
    dpath = data_path(file_name)
    with open(dpath, 'wb') as file:
        dump(profile, file)
    print(
        "* INFO:",
        file_name,
        "training data is successfully saved\n",
        file=stdout)


def fetch_pca(data: tuple) -> tuple:
    profile = build_pca(data)
    make_file("PCA", profile)
    return profile


def fetch_svm(data: tuple, eigen: bool = True) -> tuple:
    profile = build_svm(data, eigen)
    file_name = "SVM"
    if eigen:
        file_name = "PCAn" + file_name
    make_file(file_name, profile)
    return profile


def fetch_model(algorithm: str = "ALL", eigen: bool = True):
    x, y, names = load_profile()
    m, n = x.shape
    n_classes = len(names)
    if K > m:
        k = m
    else:
        k = K
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE)
    print(f"\n{m} samples from {n_classes} people are loaded\n")
    print("Samples:", m)
    print("Features:", n)
    print("Classes:", n_classes, "\n")
    data = (x_train, x_test, y_train, y_test, k, names)
    if algorithm == "PCA":
        return fetch_pca(data)
    if algorithm == "SVM" or "PCAnSVM":
        return fetch_svm(data, eigen)
    if algorithm == "ALL":
        fetch_pca(data)
        fetch_svm(data)
        if not eigen:
            fetch_svm(data, False)


def name_pca(face: list) -> str:
    file_name = "PCA"
    cache = data_path(file_name)
    if cache.exists():
        with open(cache, 'rb') as file:
            y_train, u, omega, names = load(file)
    else:
        y_train, u, omega, names = fetch_model(file_name)
    img = fetch_image(face)
    img = img_as_float(img).astype(np.float32)
    test_data = np.array(img).reshape(1, -1)
    idx = pred_pca(test_data, u, omega)
    pred = y_train[idx]
    name = names[np.int(pred)]
    return name


def name_svm(face: list, eigen: bool = True) -> str:
    file_name = "SVM"
    if eigen:
        file_name = "PCAn" + file_name
    cache = data_path(file_name)
    if cache.exists():
        with open(cache, 'rb') as file:
            pca, clf, names = load(file)
    else:
        pca, clf, names = fetch_model(file_name, eigen)
    img = fetch_image(face)
    test_data = np.array(img).reshape(1, -1)
    if eigen:
        test_data = pca.transform(test_data)
    pred = clf.predict(test_data)
    name = names[np.int(pred)]
    return name
