# 🐕 Global Dog Breed Identification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **An end-to-end Deep Learning application for identifying dog breeds from images using MobileNetV2 Transfer Learning, with Dog/Non-Dog pre-screening and an interactive Streamlit interface.**

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [How the System Works](#-how-the-system-works)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture)
- [Dog / Non-Dog Pre-Screening](#-dog--non-dog-pre-screening)
- [MobileNetV2 Breed Classifier](#-mobilenetv2-breed-classifier)
- [Training Strategy](#-training-strategy)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Data Augmentation](#-data-augmentation)
- [Dataset Split](#-dataset-split)
- [Class Mapping](#-class-mapping)
- [Model Performance](#-model-performance)
- [Application](#-application)
- [Application Workflow](#-application-workflow)
- [Breed Profile & History Database](#-breed-profile--history-database)
- [Repository Structure](#-repository-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Docker](#-docker)
- [Example Usage](#-example-usage)
- [Project Highlights](#-project-highlights)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

# 📖 About the Project

**Global Dog Breed Identification** is a Deep Learning-based computer vision project designed to identify dog breeds from images.

The system combines:

- Computer Vision
- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2
- Image Classification
- PyTorch
- Streamlit

The project supports **356 dog-breed classes** within the project's dataset and trained model.

Instead of directly classifying every uploaded image, the application uses a two-stage workflow:

```text
                    Uploaded Image
                          │
                          ▼
              ┌──────────────────────┐
              │ Dog / Non-Dog Check  │
              └──────────────────────┘
                    │            │
                 Non-Dog         Dog
                    │            │
                    ▼            ▼
                Warning      MobileNetV2
                                 │
                                 ▼
                       356-Breed Classifier
                                 │
                                 ▼
                         Top-3 Predictions
                                 │
                                 ▼
                    Breed Profile & History