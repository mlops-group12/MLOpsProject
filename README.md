# Convolutional Neural Network classifying objects in CIFAR-10

This repository contains the implementation for our Machine learning operations project at DTU (course 02476), where we implement a Convolutional Neural Network (CNN) capable of classifying objects in images using the CIFAR-10 dataset. Our code and experiments are included in the repository. This project is built using the cookie-cutter template as a standardized structured framework to run the project as simply as possible.

## Authors
| Name | Student ID |
|------|------------|
| Jacob Borregaard Eriksen | s181487 |
| Liv Dreyer Johansen | s214613 |
| Nikolaj Hertz | s214644 |
| Signe Djernis Olsen | s206759 |

## Overall Goal of the Project
The goal of this project is to design, implement and evaluate a Convolutional Neural Network capable of classifying objects in images. The model will take an image as input and output a ten-dimensional probability vector that represents the likelihood of the image belonging to each of the predefined classes. The CIFAR-10 dataset will be used as the primary benchmark, allowing the model to learn meaningful visual features and distinguish between common object categories such as airplanes, cars, cats, dogs and ships.

## Framework
The project will be implemented using PyTorch as the core deep learning library. To improve code structure and reduce boilerplate code related to training and validation loops, PyTorch Lightning will be used as a high-level framework built on top of PyTorch. PyTorch Lightning will be integrated into the project to manage training, validation and logging while preserving the underlying PyTorch model definitions.

## Data
The project will use the CIFAR-10 dataset, which consists of 60.000 color images of size 32×32 pixels divided into 10 distinct classes. The dataset is split into 40.000 training images, 10.000 validation images and 10.000 test images.

## Model
The Convolutional Neural Network will be based on common computer vision principles. Initially, the data is passed through several convolutional blocks, including batch-normalization and ReLU. At appropriate places, the blocks will be joined by pooling layers. After this feature extraction, a fully connected network will follow. This will output the aforementioned 10-dimensional output vector, allowing us to obtain class probabilities.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   └── cifar-10
        └── cifar-10-baches-py
            ├── data_batch_1
            ├── data_batch_2
            ├── data_batch_3
            ├── data_batch_4
            └── data_batch_5
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
