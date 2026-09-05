# 🐕 Global Dog Breed Identification Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system that classifies dog images across **356 distinct dog breed classes** using Convolutional Neural Networks (CNNs) and Transfer Learning (**MobileNetV2**). 

The project features a **two-stage pipeline**: an automated **Dog / Non-Dog pre-screening filter** followed by fine-grained **356-breed classification**. Predictions are paired with an interactive **Breed Profile & History Database** across 15 detailed fields, accessible through a modern, responsive **Streamlit Web Application**.

---

## 📌 Project Overview

Identifying dog breeds accurately from photographs is a fine-grained computer vision challenge due to high inter-class visual similarity (e.g., Belgian Malinois vs. German Shepherd) and high intra-class variation across ages, coat colors, and poses. 

This project solves fine-grained dog breed classification at scale by:
1. **Filtering out non-dog uploads** (humans, cats, cars, objects) using a pre-screening vision model before breed prediction.
2. **Classifying 356 FCI-recognized dog breeds** with a high-accuracy MobileNetV2 transfer learning model.
3. **Retrieving full breed profiles & history** (origin, lifespan, temperament, exercise needs, health risks, historical evolution) for every predicted breed.

---

## ✨ Key Features

* **356 Dog Breed Scope:** Full coverage of 356 recognized dog breeds matching the **Fédération Cynologique Internationale (FCI)** standards.
* **Pre-Screening Dog / Non-Dog Detector:** Pre-screens uploaded images to ensure only canine images proceed to breed classification, preventing false predictions on non-dog inputs (cats, humans, vehicles, etc.).
* **MobileNetV2 Transfer Learning:** Pretrained backbone fine-tuned for 356-class categorization, reaching **72.47% Top-1 Accuracy** and **86.05% Top-5 Accuracy** on test datasets.
* **Top-3 Confidence Breakdown:** Displays primary predictions along with visual progress metrics for the top 3 candidate breeds.
* **Automated 15-Field Breed Profile & History:** Instantly loads comprehensive breed profiles:
  1. *Breed Name*
  2. *Group / Type*
  3. *Country of Origin*
  4. *Original Purpose*
  5. *Average Lifespan*
  6. *Size Category*
  7. *Average Height*
  8. *Average Weight*
  9. *Temperament*
  10. *Breed History & Evolution*
  11. *Exercise Needs*
  12. *Grooming Needs*
  13. *Suitable Living Environment*
  14. *Common Health Information*
  15. *Interesting Facts*
* **Interactive 356-Breed Catalog Explorer:** Allows users to search and read full history profiles for all 356 breeds without uploading an image.
* **Modern Streamlit UI/UX:** Responsive, dark/light compatible web application with custom CSS result cards, upload reset options, and clear visual hierarchy.

---

## 🧠 Model Architecture & Performance

### 1. Pre-Screening Dog / Non-Dog Detector
- **Architecture:** Lightweight Vision Classifier (ImageNet backbone feature extractor).
- **Function:** Evaluates input image logits for canine visual feature signatures.
- **Behavior:** If non-canine features dominate, classification is safely halted with a clear warning notification to the user.

### 2. 356-Breed Classifier (MobileNetV2 Transfer Learning)
- **Base Model:** MobileNetV2 pretrained on ImageNet.
- **Classifier Head:** Dropout (0.3) + Linear Output Layer (356 output logits).
- **Input Resolution:** $224 \times 224$ RGB.
- **Optimization:** Adam Optimizer ($lr = 1 \times 10^{-4}$ with Cosine Annealing scheduler), Cross-Entropy Loss.
- **Performance Summary:**
  - **Top-1 Test Accuracy:** `72.47%`
  - **Top-5 Test Accuracy:** `86.05%`
  - **Total Parameters:** 3,902,564

---

## 📊 Dataset & Breed Coverage

- **Total Classes:** 356 Dog Breeds (1 breed = 1 index, `0` to `355`).
- **Class Mapping File:** [`data/processed/dog_breeds_356.json`](data/processed/dog_breeds_356.json).
- **Dataset Size:** 7,120 decodable RGB dog images (20 verified images per breed).
- **Data Splits:** 70% Train / 15% Validation / 15% Test.
- **Data Augmentation:** Random Resized Crops, Horizontal Flips, Random Rotations ($15^\circ$), and Color Jitter.

---

## 💻 Streamlit Web Application

The interactive web UI is located in `app/app.py` and provides two primary workflows:

1. **📷 Upload & Identify Image:**
   - Drag-and-drop dog photo upload.
   - Live image resolution and format detection.
   - Pre-screening dog filter check.
   - Prominent prediction card with confidence percentage badge.
   - Top-3 medal rankings (`🥇 #1`, `🥈 #2`, `🥉 #3`) with progress bars.
   - Full centered 2-column Breed Profile & History section for the predicted dog.
   - One-click **Try Another Image** reset.

2. **🔍 Search 356 Breed Information:**
   - Dropdown and search box to explore profiles for any of the 356 dog breeds instantly.

---

## 📁 Repository Structure

```
image-classification-cnn/
├── app/
│   ├── app.py                      # Streamlit Web Application (UI/UX)
│   └── utils.py                    # Image validation & loading utilities
├── data/
│   ├── breed_information/
│   │   └── dog_breed_info.json     # 15-field profile DB for 356 dog breeds
│   ├── processed/
│   │   ├── dataset_splits.json     # Train/val/test splits mapping
│   │   └── dog_breeds_356.json     # Exact 356 class names (indices 0-355)
│   └── test_samples/               # Sample evaluation images
├── docs/                           # Documentation assets
├── models/
│   ├── dog_breed_transfer.pt       # Trained 356-class MobileNetV2 model (Production)
│   ├── cnn_transfer.pt             # Transfer model checkpoint
│   └── cnn_scratch.pt              # Baseline CNN scratch checkpoint
├── notebooks/                      # Development & exploration notebooks
├── results/
│   ├── class_distribution.png      # Dataset distribution chart
│   ├── dataset_summary.txt         # Dataset verification summary
│   ├── model_comparison.json       # Training & evaluation metric results
│   └── sample_images.png           # Visual sample grid
├── src/
│   ├── data_loader.py              # PyTorch Dataset & DataLoader pipelines
│   ├── dog_detector.py             # Pre-screening Dog / Non-Dog detection module
│   ├── evaluate.py                 # Evaluation pipeline (Top-1, Top-5, Confusion Matrix)
│   ├── model.py                    # PyTorch Neural Network architectures
│   ├── predict.py                  # Inference pipeline & model loader
│   ├── train.py                    # Model training loop & checkpoint saver
│   └── utils.py                    # Preprocessing & metric calculation helpers
├── .dockerignore                   # Docker exclusion patterns
├── .gitignore                      # Git exclusion rules
├── Dockerfile                      # Containerization instructions
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/image-classification-cnn.git
cd image-classification-cnn
```

### 2. Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

To start the Streamlit web application:

```bash
streamlit run app/app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## 🐳 Docker Deployment (Optional)

To build and run the application in a Docker container:

```bash
# Build the Docker image
docker build -t dog-breed-identifier .

# Run the container
docker run -p 8501:8501 dog-breed-identifier
```

Then access the application at `http://localhost:8501`.

---

## 📄 License & Disclaimer

* **Disclaimer:** *This application is intended for educational, research, and informational purposes. Breed profiles, health recommendations, and care guidance do not replace professional veterinary diagnostics.*
* **License:** Distributed under the [MIT License](LICENSE).
