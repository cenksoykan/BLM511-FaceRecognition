"""
script.py
"""

from functools import lru_cache

from pred import live_svm
from train import add_profile
from config import DATA_FOLDER, FACE
from helper.data import name_pca, name_svm, fetch_model
from helper.image import check_path

PRETEXT = "\nRunning Task"
ALGORITHMS = ["Eigenface", "SVM", "PCA + SVM"]
TEST_DATA = DATA_FOLDER.joinpath(FACE.PROFILE, FACE.IMG)


def invalid_choice():
    while True:
        try:
            choice = int(input(">>> Please enter a valid choice: ").strip())
        except ValueError:
            continue
        else:
            choose_task(choice)


@lru_cache(maxsize=16)
def choose_task(choice: int, done: bool = False):
    if choice == 1:
        print(PRETEXT, str(choice) + ": Training")
        fetch_model()
    elif choice in (2, 3, 4):
        print(PRETEXT, str(choice) + ":", ALGORITHMS[choice - 2])
        test_face = check_path(TEST_DATA)
        pred_name = "UNKNOWN"
        if choice == 2:
            pred_name = name_pca(test_face)
        elif choice == 3:
            pred_name = name_svm(test_face, False)
        elif choice == 4:
            pred_name = name_svm(test_face)
        print(f"* RESULT: This is picture of {pred_name!r}")
    elif choice == 5:
        print(PRETEXT, str(choice) + ": Real-time", ALGORITHMS[2])
        live_svm()
    elif choice == 6:
        print(PRETEXT, str(choice) + ": Add face profile")
        profile_name = input(
            "\n>>> Please enter a profile name: ").strip().capitalize()
        if profile_name:
            add_profile(profile_name)
    elif choice == 10:
        print(PRETEXT, str(choice) + ": Train all")
        fetch_model(eigen=False)
    elif choice == 0:
        quit()
    else:
        invalid_choice()
    if done:
        quit()
    task_selector()


def task_selector():
    print("\n   TASKS\n"
          "1: Training\n"
          "2: Eigenface\n"
          "3: SVM *\n"
          "4: PCA + SVM\n"
          "5: Real-time recognition\n"
          "6: Add a face profile\n"
          "0: Quit\n"
          "\n\t* Training may take a long runtime on the very first!"
          "\n\t  To train it with others use task 10.\n")
    try:
        choice = int(input(">>> Enter your choice: ").strip())
    except ValueError:
        invalid_choice()
    choose_task(choice)
