"""
build.py
"""

from time import perf_counter
from sys import stdout

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from cv2 import imwrite

from config import VERBOSE, SVM, FACE_DIM, THRESHOLD
from helper.image import check_format
from helper.profile import create_profile


def print_time(t_0: float) -> float:
    t = perf_counter() - t_0
    print(f"\t-> done in {t:.3f}s")
    return t


def success_percent(prediction: list, actual: list, algorithm: str):
    if prediction.shape == actual.shape:
        success = np.sum(prediction == actual) / len(actual) * 100
        print(algorithm, f"Error Rate:\t\t{100 - success:.4f}%")
        print(algorithm, f"Recognition Rate:\t{success:.4f}%\n")


def norm_img(image, um: int = 150, ustd: int = 120):
    S = image.T
    S -= np.mean(S, axis=0)
    S /= np.std(S, axis=0)
    S *= ustd
    S += um
    return S.T


def mean_img(image):
    m = np.mean(image, axis=0)
    m = m.astype(np.uint8)
    return m


def norm_eig(eig):
    for i in range(eig.shape[1]):
        v_i = eig[:, i]
        kk = v_i**2
        kk_sum = np.sum(kk)
        eig[:, i] = v_i / np.math.sqrt(kk_sum)
    return eig


def save():
    valid = {"yes": True, "y": True, "no": False, "n": False, "": False}
    check = "\n>>> Do you want to add in a different profile [y/N]: "
    while True:
        choice = input(check).strip().lower()
        if choice in valid:
            return valid[choice]
        print("\n\t! Please respond with 'yes' or 'no'\n")


def save_img(image):
    image = image.reshape(-1, FACE_DIM[0])
    plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.show()
    if save():
        print("\n* INFO: Data must be retrained\n", file=stdout)
        profile_name = input("\n>>> Profile name: ").strip().capitalize()
    else:
        profile_name = None
    if profile_name:
        profile_folder = create_profile(profile_name)
        index = 1
        for picture in list(profile_folder.iterdir()):
            if check_format(picture):
                index += 1
        file_name = profile_name + "-" + str(index) + ".pgm"
        new_face = profile_folder.joinpath(file_name)
        imwrite(new_face, image)
        print(
            f"\n* INFO: New {profile_name!r} profile created\n"
            "\n* WARNING: Data must be retrained\n",
            file=stdout)
    else:
        print(f"\n* INFO: Image not renamed as a new profile\n", file=stdout)


def weight_pca(x_test, u):
    S = norm_img(x_test)
    A = S.T
    w = np.dot(u.T, A).astype(np.float32)  # [k x M_i]
    return w


def pred_pca(x_test, u, omega):
    w = weight_pca(x_test, u)
    diff = omega - w  # [k x M]
    norms = np.linalg.norm(diff, axis=0)  # [M]
    if VERBOSE:
        norms_min = np.min(norms)
        threshold = THRESHOLD
        if norms_min > threshold:
            print(
                "\n* INFO: A new face was found\t->",
                norms_min,
                ">",
                threshold,
                file=stdout)
            save_img(x_test)
    idx = np.argmin(norms)
    return idx


def build_pca(data: tuple) -> tuple:
    x_train, x_test, y_train, y_test, k, names = data
    x_train = img_as_float(x_train).astype(np.float32)
    x_test = img_as_float(x_test).astype(np.float32)

    t_1 = 0
    print("Normalizing the data", end="")
    t_0 = perf_counter()
    # Normalize all images to reduce the error due to lighting conditions
    S = norm_img(x_train)
    t_1 += print_time(t_0)

    if VERBOSE:
        m = mean_img(S)  # Mean image
        # Reshape flattened image with its dimension and show
        img = m.reshape(-1, FACE_DIM[0])
        io.imshow(img)  # Show mean image
        io.show()

    A = S.T  # [N^2 x M]
    print("Calculating covariance", end="")
    t_0 = perf_counter()
    # Covariance matrix C=AA', L=A'A
    # C = np.dot(A, A.T)  # [N^2 x N^2]
    L = np.dot(A.T, A)  # [M x M]
    L /= A.shape[1]  # divide by M
    t_1 += print_time(t_0)

    print("Finding eigenvalues and eigenvectors", end="")
    t_0 = perf_counter()
    # vv are the eigenvector for L; dd are the eigenvalue for both L and C
    dd, vv = np.linalg.eig(L)  # [M], [M x M]
    t_1 += print_time(t_0)

    for ev in vv:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    if VERBOSE:
        with plt.style.context('seaborn-whitegrid'):
            x_range = range(1, len(dd) + 1)
            tot = sum(dd)
            dd_desc = sorted(dd, reverse=True)
            var_exp = [100 * i / tot for i in dd_desc]
            cum_var_exp = np.cumsum(var_exp)
            plt.figure(figsize=(9, 6))
            plt.bar(
                x_range,
                var_exp,
                alpha=0.5,
                align='center',
                label='individual explained variance')
            plt.step(
                x_range,
                cum_var_exp,
                where='mid',
                label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.ylim(0, 100)
            plt.xlim(0, k * 1.5)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

    print("Selecting principal components", end="")
    t_0 = perf_counter()
    # Sort the eigen values in descending order and then sorted the eigen vectors by the same index
    idx = (-dd).argsort()[:k]  # [k]
    # d = dd[idx]  # [k]
    v = vv[:, idx]  # [M x k]
    v = norm_eig(v)  # Normalization of eigenvectors

    # Eigenvectors of C matrix
    u = np.dot(A, v).astype(np.float32)  # [N^2 x k]
    u = norm_eig(u)  # Normalization of eigenvectors

    # Find the weight of each face in the training set.
    omega = np.dot(u.T, A).astype(np.float32)  # [k x M]
    t_1 += print_time(t_0)

    y_pred = []
    print("Predicting people's names on the test set", end="")
    t_0 = perf_counter()
    for i in range(len(x_test)):
        test_i = np.array(x_test[i, :]).reshape(1, -1)
        idx = pred_pca(test_i, u, omega)
        prediction = y_train[idx]
        y_pred.append(prediction)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    t = perf_counter() - t_0
    print(f"\t-> took {1000 * t / len(y_pred):.4f}ms per sample on average")
    t_1 += t

    print(f"\n\t=> PCA was completed in {t_1:.3f}s\n")

    success_percent(y_pred, y_test, "PCA")
    return y_train, u, omega, names


def build_svm(data: tuple, eigen: bool = True) -> tuple:
    x_train, x_test, y_train, y_test, k, names = data
    test_data = list(x_test)
    t_1 = 0
    if eigen:
        print(
            f"Extracting the top {k} eigenfaces from {len(x_train)} faces",
            end="")
        t_0 = perf_counter()
        pca = PCA(
            n_components=k, svd_solver='randomized', whiten=True).fit(x_train)
        t_1 += print_time(t_0)

        print(
            "Projecting the input data on the eigenfaces orthonormal basis",
            end="")
        t_0 = perf_counter()
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        t_1 += print_time(t_0)
    else:
        pca = None

    print("Fitting the classifier to the training set", end="")
    t_0 = perf_counter()
    param_grid = {
        'C': SVM.C,
        'gamma': SVM.GAMMA,
    }
    clf = GridSearchCV(
        SVC(kernel=SVM.KERNEL, class_weight='balanced'), param_grid)
    # clf = GridSearchCV(
    #     SVC(kernel='rbf',
    #         class_weight='balanced',
    #         cache_size=200,
    #         coef0=0.0,
    #         decision_function_shape='ovr',
    #         degree=3,
    #         max_iter=-1,
    #         probability=False,
    #         random_state=None,
    #         shrinking=True,
    #         tol=0.001,
    #         verbose=False), param_grid)
    clf = clf.fit(x_train, y_train)
    t_1 += print_time(t_0)
    # print("Best estimator found by grid search:", clf.best_estimator_)

    print("Predicting people's names on the test set", end="")
    t_0 = perf_counter()
    y_pred = clf.predict(x_test)
    t = perf_counter() - t_0
    print(f"\t-> took {1000 * t / len(y_pred):.4f}ms per sample on average")
    t_1 += t

    print(f"\n\t=> SVM was completed in {t_1:.3f}s\n")

    # print(
    #     classification_report(y_test, y_pred, target_names=names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    success_percent(y_pred, y_test, "SVM")

    if VERBOSE:
        prediction_titles = [
            title(names, y_pred[i], y_test[i]) for i in range(len(y_pred))
        ]
        plot_gallery(test_data, prediction_titles)
        plt.show()
    return pca, clf, names


def plot_gallery(images, titles, n_row: int = 3, n_col: int = 4):
    plt.figure(figsize=(1.5 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(FACE_DIM), cmap='gray')
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()


def title(target_names: list, pred_i: int, test_i: int) -> str:
    pred_name = target_names[pred_i]
    true_name = target_names[test_i]
    return f"pred: {pred_name}\ntest: {true_name}"
