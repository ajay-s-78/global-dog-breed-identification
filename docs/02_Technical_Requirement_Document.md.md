# Technical Requirement Document (TRD)
## Image Classification using CNNs

---

## 1. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Language** | Python 3.10+ | Standard for ML/DL work, huge ecosystem |
| **Deep Learning Framework** | TensorFlow/Keras **or** PyTorch (pick one) | Both support CNNs well; Keras is beginner-friendlier, PyTorch is more flexible/industry-preferred for research |
| **Data Handling** | NumPy, Pandas, OpenCV/Pillow | Image loading, array manipulation |
| **Visualization** | Matplotlib, Seaborn | Plotting accuracy/loss curves, confusion matrix |
| **Model Training Env** | Google Colab (free GPU) or local GPU | CNN training is much faster on GPU |
| **Web App Framework** | Streamlit (simplest) or Flask (more control) | Streamlit = fastest for ML demos; Flask = more customizable, closer to real backend work |
| **Containerization** | Docker | Makes the app portable/runnable anywhere |
| **Cloud (Optional)** | AWS (EC2/S3) or GCP (Compute Engine/Cloud Storage) | For public deployment |
| **Version Control** | Git + GitHub | Track code, show project publicly |

**Recommendation for a first DL project:** Use **TensorFlow/Keras** (simpler syntax, faster to get results) + **Streamlit** (drag-and-drop UI in minutes) + **Docker** (basic). Skip AWS/GCP unless time permits in week 2.

---

## 2. Dataset Requirements

| Item | Detail |
|---|---|
| **Recommended datasets** | CIFAR-10 (10 classes, easy, built into TensorFlow/PyTorch) OR Intel Image Classification (6 classes: buildings, forest, glacier, mountain, sea, street) OR a custom Kaggle dataset (e.g., cats vs dogs) |
| **Size** | 5,000–60,000 images depending on dataset (CIFAR-10 has 60,000) |
| **Format** | JPG/PNG, resized to a fixed size (e.g., 32x32 or 224x224) |
| **Split** | 70% train / 15% validation / 15% test (or use dataset's built-in split) |
| **Labels** | Categorical, single-label per image |

**Term explained — "single-label classification":** each image belongs to exactly ONE category (a photo is either "cat" OR "dog", not both). This is different from multi-label, where one image could have several tags.

---

## 3. Model Requirements

### 3.1 Architecture Options

| Approach | Description | When to Use |
|---|---|---|
| **CNN from scratch** | Build your own Conv2D + Pooling + Dense layers | To learn CNN fundamentals deeply |
| **Transfer Learning** | Use a pretrained model (MobileNetV2, ResNet50, VGG16) and fine-tune the last layers | To get high accuracy fast with less data/compute |

**Recommended plan:** Build ONE CNN from scratch first (for learning + to show in interviews you understand the internals), THEN do transfer learning with MobileNetV2 (lightweight, fast, good accuracy) and compare results. This comparison itself becomes a strong talking point in interviews.

### 3.2 Key Model Components (terms explained)

- **Convolution Layer (Conv2D):** Slides small filters over the image to detect patterns like edges, textures, shapes.
- **Pooling Layer (MaxPooling):** Shrinks the image size while keeping important features — reduces computation and overfitting.
- **Activation Function (ReLU):** Adds non-linearity so the network can learn complex patterns, not just straight lines.
- **Flatten Layer:** Converts the 2D feature maps into a 1D vector before feeding into Dense layers.
- **Dense (Fully Connected) Layer:** Makes the final decision — outputs probabilities for each class.
- **Softmax:** Final activation that converts raw outputs into probabilities that sum to 1 (used for multi-class classification).
- **Dropout:** Randomly "turns off" some neurons during training to prevent overfitting (model memorizing instead of learning).

### 3.3 Training Configuration

| Parameter | Suggested Value | Notes |
|---|---|---|
| Optimizer | Adam | Fast, works well by default |
| Loss Function | Categorical Crossentropy (or Sparse Categorical Crossentropy) | Standard for multi-class classification |
| Batch Size | 32 or 64 | Balance between speed and memory |
| Epochs | 15–30 (with early stopping) | Avoid overfitting/underfitting |
| Learning Rate | 0.001 (default Adam) | Can reduce if loss doesn't improve |
| Data Augmentation | Rotation, flip, zoom, shift | Increases effective dataset size, improves generalization |

---

## 4. Evaluation Requirements

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall % of correct predictions |
| **Precision** | Of predicted "cats", how many were actually cats? |
| **Recall** | Of actual cats, how many did we correctly find? |
| **F1-Score** | Balance between precision and recall |
| **Confusion Matrix** | Visual grid showing which classes get confused with which |
| **Loss/Accuracy Curves** | Training vs validation curves to check for overfitting |

---

## 5. Application (Inference) Requirements

- Accept image upload (JPG, PNG; max size ~5MB).
- Preprocess uploaded image to match training input shape (resize + normalize).
- Load saved model (`.h5` for Keras or `.pt` for PyTorch) once at app startup (not on every request — for speed).
- Return prediction + confidence score.
- Handle errors: wrong file type, corrupted image, no file selected.

---

## 6. Non-Functional / System Requirements

| Requirement | Detail |
|---|---|
| **Hardware (training)** | GPU recommended (Colab free T4 GPU is sufficient) |
| **Hardware (inference/deployment)** | CPU is enough — models like MobileNetV2 are lightweight |
| **Model size** | Keep under ~50MB for fast loading in the app |
| **Response time** | < 3 seconds per prediction on CPU |
| **Docker image size** | Keep reasonably small; use slim Python base image |

---

## 7. File/Folder Structure (Technical Standard)

```
image-classification-cnn/
│
├── data/
│   ├── raw/                # original downloaded dataset
│   └── processed/          # resized/augmented data (if saved separately)
│
├── notebooks/
│   └── exploration.ipynb   # EDA and experimentation
│
├── src/
│   ├── data_loader.py      # loading & preprocessing functions
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # training script
│   ├── evaluate.py         # evaluation & metrics
│   └── predict.py          # inference function
│
├── models/
│   └── cnn_model.h5        # saved trained model
│
├── app/
│   ├── app.py               # Streamlit or Flask app
│   └── utils.py              # image preprocessing for inference
│
├── Dockerfile
├── requirements.txt
├── README.md
└── docs/                     # all 6 project documents
```

---

## 8. Tools/Libraries Summary Table

| Purpose | Library |
|---|---|
| Deep Learning | tensorflow / torch |
| Image Processing | opencv-python, Pillow |
| Data Handling | numpy, pandas |
| Visualization | matplotlib, seaborn |
| Web App | streamlit / flask |
| Metrics | scikit-learn |
| Containerization | docker |
