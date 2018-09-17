"""
========================
    Face recognition
========================

Summary:
    Real time face tracking and recognition using Principal Component Analysis (PCA) and Support Vector Machine (SVM) classifier
"""

from sys import argv, stderr

from script import choose_task, task_selector

print(__doc__)


def main():
    if len(argv) == 2:
        choose_task(int(argv[1]), True)
    elif len(argv) > 2:
        print("\n* ERROR: More than one argument specified\n", file=stderr)
        exit()
    task_selector()


if __name__ == "__main__":
    main()
