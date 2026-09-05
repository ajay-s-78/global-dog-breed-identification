# App Flow Document
## Image Classification using CNNs

---

## 1. High-Level Flow (End-to-End)

```
[User] 
   |
   v
[Opens Web App (Streamlit/Flask)]
   |
   v
[Uploads an Image] 
   |
   v
[Frontend sends image to backend/inference function]
   |
   v
[Image Preprocessing: resize, normalize, convert to array]
   |
   v
[Trained CNN Model loaded in memory]
   |
   v
[Model runs prediction on the image]
   |
   v
[Output: predicted class + confidence score]
   |
   v
[Result displayed back to user on screen]
```

---

## 2. Two Separate Flows in This Project

It's important to understand this project has **two independent pipelines**:

### Flow A — Model Training Flow (happens ONCE, offline, before deployment)

```
Raw Dataset
   |
   v
Data Cleaning & Splitting (train/val/test)
   |
   v
Data Augmentation (rotate, flip, zoom)
   |
   v
CNN Model Building (define layers)
   |
   v
Model Training (fit on train data, validate on val data)
   |
   v
Model Evaluation (test data → accuracy, confusion matrix)
   |
   v
Save Model (cnn_model.h5)
```

### Flow B — Inference/App Flow (happens EVERY TIME a user uploads an image, in real time)

```
User uploads image
   |
   v
App loads saved model (only once, at startup)
   |
   v
Image preprocessed to match model's expected input
   |
   v
model.predict(image) 
   |
   v
Get class with highest probability + confidence %
   |
   v
Display result on screen
```

**Why this separation matters:** Training is slow and done once on powerful hardware (GPU). Inference (using the trained model to predict) is fast and must run instantly for the user — this is a core concept in real ML systems (train once, deploy many times).

---

## 3. Detailed Step-by-Step User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | User opens the app in browser | App loads with title, instructions, and an upload button |
| 2 | User clicks "Browse" or drags an image | App validates file type (jpg/png only) |
| 3 | User confirms upload | App shows a preview of the uploaded image |
| 4 | (Automatic) | App preprocesses image (resize to model's input size, e.g. 224x224, normalize pixel values 0–1) |
| 5 | (Automatic) | Model runs prediction |
| 6 | (Automatic) | App displays: **Predicted Class**, **Confidence %**, optionally a bar chart of top-3 class probabilities |
| 7 | User can upload another image | Flow repeats from Step 2 |

---

## 4. Error Handling Flow

```
User uploads file
   |
   v
Is it a valid image file (jpg/png)?
   |--- No --> Show error: "Please upload a valid JPG/PNG image"
   |
  Yes
   |
   v
Is file size within limit (e.g. <5MB)?
   |--- No --> Show error: "File too large, please upload under 5MB"
   |
  Yes
   |
   v
Proceed to preprocessing & prediction
```

---

## 5. Architecture-Level Flow (System View)

```
 ┌─────────────┐        ┌──────────────────┐        ┌─────────────────┐
 │   Browser    │  --->  │   Web App Layer   │  --->  │  Inference Layer │
 │ (User Upload)│        │ (Streamlit/Flask) │        │  (Loaded CNN)    │
 └─────────────┘        └──────────────────┘        └─────────────────┘
                                   |                          |
                                   v                          v
                          Preprocessing Logic          Trained Model File
                          (resize/normalize)               (cnn_model.h5)
```

---

## 6. Docker Flow (Deployment)

```
Dockerfile defines environment
   |
   v
docker build -> creates image with Python + libraries + app code + model
   |
   v
docker run -> starts container, exposes port (e.g. 8501 for Streamlit)
   |
   v
User accesses app via localhost:8501 (or public IP if deployed to cloud)
```

---

## 7. Key Takeaway

Think of the whole system as two clocks running at different speeds:
- **Slow clock (once):** Data → Train → Evaluate → Save Model.
- **Fast clock (every request):** Upload → Preprocess → Predict → Show Result.

Keeping these two flows mentally separate is one of the most important habits for building real ML applications — and a common interview question is exactly this: *"Explain your training pipeline vs your inference pipeline."*
