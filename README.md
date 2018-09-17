# BLM511-FaceRecognition

Edit variables in [config.py][vars] file under `./face-recognition/` directory.

## Setup

Before setup make sure that you have correct version of python. If you have installed different version of python on your system you may want to use [pyenv] for install any specific version.

There are two suggested ways to run this application. In case you choose the solution with [pipenv] setup then it will install required python automatically if [pyenv] is available.

### Setup and run with pipenv package

Learn more details about [pipenv]

Install pipenv

```sh
pip install pipenv
```

Install dependencies

```sh
pipenv install
```

Run the application

```sh
pipenv run python face-recognition
```

### Setup and run with pip

Learn more details about [pyenv]

Create virtual environment

```sh
python3 -m venv ./.venv
```

Activate virtual environment

```sh
source ./.venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```sh
python face-recognition
```

Deactivate virtual environment

```sh
deactivate
```

## Description of Data and Source

[The extended Yale Face Database B][yale] contains 16128 images of 28 human subjects under 9 poses and 64 illumination conditions. The data format of this database is the same as the Yale Face Database B.

> Georghiades, A., Belhumeur, P., & Kriegman, D. (2001). From few to many: Illumination cone models for face recognition under variable lighting and pose. _IEEE Trans. Pattern Anal. Mach. Intelligence_, _23_(6), 643–660.

[vars]: ./face-recognition/config.py
[pyenv]: https://github.com/pyenv/pyenv "Simple Python Version Management"
[pipenv]: https://github.com/pypa/pipenv "Python Development Workflow for Humans"
[yale]: http://vision.ucsd.edu/content/extended-yale-face-database-b-b "Extended Yale Face Database B (B+)"
