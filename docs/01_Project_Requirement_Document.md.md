# Project Requirement Document (PRD)
## Image Classification using CNNs — 2 Week Deep Learning Project

---

## 1. Project Overview

**Project Title:** Image Classification using Convolutional Neural Networks (CNNs)

**Duration:** 2 Weeks

**Type:** Deep Learning / Computer Vision Project (End-to-End)

**One-line description:** Build a CNN model that takes an image as input and predicts what category it belongs to (e.g., cat, dog, car, flower), then deploy it as a working web app so anyone can upload an image and get a live prediction.

---

## 2. Problem Statement

Manually sorting or tagging large volumes of images (products, medical scans, plants, animals, etc.) is slow and error-prone. We need an automated system that can look at an image and correctly classify it into a predefined category, with reasonable speed and accuracy, and expose this capability through a simple web interface.

---

## 3. Project Goals

| Goal | Description |
|---|---|
| **Primary Goal** | Train a CNN model that classifies images into fixed categories with good accuracy (target: 85%+ on test data for a standard dataset). |
| **Secondary Goal** | Wrap the model in a usable web app (Streamlit or Flask) where a user uploads an image and instantly sees the predicted class + confidence score. |
| **Stretch Goal** | Containerize with Docker and optionally deploy to AWS/GCP for public access. |

---

## 4. Why This Project (Business/Real-World Value)

- **E-commerce:** Auto-tag product images (shirt, shoe, bag) instead of manual entry.
- **Agriculture:** Classify plant leaf diseases from photos taken on a phone.
- **Healthcare:** Classify X-ray/skin images as a supporting diagnostic tool (not a replacement for doctors).
- **Wildlife/Conservation:** Identify animal species from camera-trap images.

This project simulates a real industry workflow: **data → model → API → UI → deployment**, which is exactly what companies expect from an ML/DL engineer.

---

## 5. Scope of the Project

### In Scope
- Using a public dataset (e.g., CIFAR-10, Intel Image Classification, or a custom small dataset).
- Data preprocessing and augmentation.
- Building and training a CNN from scratch AND fine-tuning a pretrained model (transfer learning) for comparison.
- Model evaluation (accuracy, confusion matrix, precision/recall).
- Saving the trained model.
- Building a simple frontend (Streamlit) or Flask + HTML app to upload an image and see predictions.
- Basic Dockerization of the app.

### Out of Scope (for this 2-week timeline)
- Training on massive datasets (millions of images) — not feasible in 2 weeks without heavy compute.
- Real-time video classification.
- Multi-label classification (only single-label, single-class-per-image).
- Building a mobile app (web app only).
- Advanced MLOps pipelines (CI/CD, model monitoring) — mentioned only as future scope.

---

## 6. Target Users

| User | Need |
|---|---|
| **End User (Demo/Recruiter)** | Wants to upload any image and instantly see what the model predicts, to judge project quality. |
| **You (Developer/Student)** | Wants a complete, resume-worthy project that proves understanding of CNNs and deployment. |
| **Future Extension: Business User** | Could use it to auto-classify product/plant/medical images in bulk. |

---

## 7. Functional Requirements

1. User can upload an image (JPG/PNG) through the web app.
2. System preprocesses the image (resize, normalize) automatically.
3. System runs the trained CNN model on the image.
4. System displays:
   - Predicted class label
   - Confidence score (%)
   - Optionally: top-3 predictions
5. System handles invalid uploads (non-image files) gracefully with an error message.
6. (Optional) User can view sample training images / class list on the app.

---

## 8. Non-Functional Requirements

| Requirement | Detail |
|---|---|
| **Performance** | Prediction should return in under 2–3 seconds on CPU. |
| **Usability** | Simple, no-login-required interface — drag & drop or click to upload. |
| **Portability** | App should run inside Docker so it works on any machine. |
| **Maintainability** | Code organized into clear modules: data, model, training, inference, app. |
| **Accuracy** | Minimum acceptable test accuracy: 80–85% depending on dataset difficulty. |

---

## 9. Success Criteria (How We Know the Project is "Done")

- [ ] Model trained and saved with test accuracy ≥ 85% (or best achievable in time given).
- [ ] Confusion matrix and evaluation metrics documented.
- [ ] Web app runs locally and correctly classifies uploaded images.
- [ ] Dockerfile builds and runs the app successfully.
- [ ] All 6 project documents completed (this document + 5 others).
- [ ] Code pushed to GitHub with a clear README.

---

## 10. Assumptions

- A public dataset will be used (no need to manually collect/label thousands of images).
- Training will be done on Google Colab (free GPU) or local machine with modest hardware.
- Deployment will be local/Docker-based by default; cloud (AWS/GCP) is optional based on time/budget.

---

## 11. Constraints

- **Time:** Strict 2-week window.
- **Compute:** Likely free-tier GPU (Colab) — limits model size and training time.
- **Team size:** Solo project (assumed).

---

## 12. Deliverables Checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | Project Requirement Document | This file |
| 2 | Technical Requirement Document | Separate file |
| 3 | App Flow | Separate file |
| 4 | UI/UX Design Brief | Separate file |
| 5 | Backend Schema | Separate file |
| 6 | Implementation Plan | Separate file |
| 7 | Trained model + code | To be built |
| 8 | Deployed/Dockerized app | To be built |
