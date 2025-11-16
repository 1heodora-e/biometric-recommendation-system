Multimodal Biometric Authentication for Personalized Product Recommendation
============================================================================

This repository contains a **Streamlit web application** that demonstrates an end‑to‑end **multimodal biometric authentication** flow combined with a **product recommendation engine**.

Users must successfully pass:
- **Facial recognition**, and
- **Voiceprint verification**

before the system runs a **personalized product recommendation model** on their profile.

The project integrates three independently trained machine learning pipelines (image, audio, tabular) into a single, production‑style app.

---

## System Demo

You can watch a short demo of the full flow (successful authentication and unauthorized attempt) here:

- **Demo video**: [Google Drive – System Demonstration](https://drive.google.com/file/d/1F-BTXwHLijkuj9gXApjubqmmkFqvrHHs/view?usp=sharing)

---

## Table of Contents

- **Project Overview**
- **Architecture & Components**
- **Model Details**
- **End‑to‑End Flow**
- **Installation & Setup**
- **Running the Application**
- **Project Structure**
- **Tech Stack**
- **Troubleshooting**
- **Team & Contributions**

---

## Project Overview

- **Goal**: Build a proof‑of‑concept system that only serves recommendations to **authenticated users**, using **face and voice** as biometric factors.
- **Domain**: Retail / e‑commerce style recommendations based on historical customer behavior and engagement.
- **Security Logic**:
  - Face must match a known user.
  - Voice must match the **same** user as the face.
  - Only then is the product recommendation model executed.

This design showcases how multimodal biometrics can be layered on top of traditional recommendation systems to improve access control and personalization.

---

## Architecture & Components

- **Frontend / Orchestration**
  - **Streamlit app** (`app.py`)
  - Handles user interaction, file uploads, control flow, and result display.

- **Image (Face) Pipeline** – `Image_processing/`
  - Preprocessing & augmentation in `Image_feature_training.ipynb`.
  - Feature extraction from face images.
  - Training a classifier for **identity recognition**.
  - Saved artifacts:
    - `image_model.pkl`
    - `image_label_encoder.pkl`

- **Audio (Voice) Pipeline** – `audio_processing/`
  - MFCC feature extraction and augmentation in notebooks.
  - Training a classifier for **speaker recognition**.
  - Saved artifacts:
    - `audio_model.pkl`
    - `audio_label_encoder.pkl`

- **Product Recommendation Pipeline** – `prediction_functionality/`
  - Data ingestion and merging:
    - `customer_social_profiles.csv`
    - `customer_transactions.csv`
  - Feature engineering and training a **RandomForestClassifier**.
  - Saved artifacts:
    - `product_recommendation_model.pkl`
    - `product_category_encoder.pkl`
    - `merged_customer_data.csv`

All three model families are loaded and orchestrated inside the Streamlit app.

---

## Model Details

- **Facial Recognition Model**
  - **Task**: Classify a face as one of four known users.
  - **Input**: 64×64 RGB face image.
  - **Features**: Flattened pixel vectors, with on‑the‑fly augmentations (rotation, shift, zoom, brightness).
  - **Model**: `RandomForestClassifier`.
  - **Dataset**: 4 users × 3 expressions (neutral, smile, surprised) plus augmentations.

- **Voiceprint Verification Model**
  - **Task**: Classify a voice sample as one of the same four users.
  - **Input**: Short audio clip (`.wav`, `.mp3`, `.flac`).
  - **Features**: 42‑dimensional **MFCC** feature vector (mean‑pooled).
  - **Model**: `RandomForestClassifier`.
  - **Augmentation**: Noise, pitch shift, time stretch, etc., scaling 8 base samples to 56 training examples.
  - **Reported Accuracy**: **91.67%** on held‑out validation.

- **Product Recommendation Model**
  - **Task**: Predict top product category recommendations for each customer.
  - **Input Features** (examples):
    - `engagement_score`
    - `purchase_interest_score`
    - `review_sentiment_encoded`
    - `avg_purchase_amount`
    - `total_spent`
    - `purchase_count`
    - `avg_rating`
  - **Model**: `RandomForestClassifier`.
  - **Reported Accuracy**: **23.08%** (limited dataset and weak feature correlation; treated as a proof‑of‑concept).

---

## End‑to‑End Flow

The Streamlit UI enforces the following sequence:

- **Step 1 – Face Verification**
  - User uploads a face image.
  - `image_model` predicts the person’s identity.
  - If the prediction is **“unauthorized”**, the app stops and denies access.

- **Step 2 – Parameter Input**
  - After a valid face, the user may enter **any parameter** (string, number, etc.).
  - This field is currently informational/logical (can be extended to tie into recommendation logic).

- **Step 3 – Voice Verification**
  - User uploads a voice sample.
  - `audio_model` predicts the speaker identity.
  - If the prediction is **“unauthorized”**, access is denied.

- **Step 4 – Product Recommendation**
  - Once both biometrics pass, the app:
    - Loads `merged_customer_data.csv`.
    - Runs `product_model.predict_proba(…)` for each customer.
    - Displays **top‑3 product categories** and associated probabilities for each customer.

---

## Installation & Setup

### Prerequisites

- **Python**: 3.9 or higher (the project was developed on Python 3.10+).
- **git**: To clone the repository.
- **OS**: Windows, macOS, or Linux.

### 1. Clone the Repository

```bash
git clone https://github.com/1heodora-e/biometric-recommendation-system.git
cd biometric-recommendation-system
```

### 2. Create and Activate a Virtual Environment

- **Windows (PowerShell / CMD)**:

```bash
python -m venv venv
.\venv\Scripts\activate
```

- **macOS / Linux**:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

The project ships with a pinned `requirements.txt`:

```bash
pip install -r requirements.txt
```

This includes:
- `streamlit`
- `scikit-learn`
- `pandas`, `numpy`
- `librosa`
- `Pillow`
- and other supporting libraries.

---

## Running the Application

With your virtual environment **activated** and the current directory set to the project root (`biometric-recommendation-system`), run:

```bash
python -m streamlit run app.py
```

Then open the URL printed to the terminal (usually `http://localhost:8501`).

> **Important**: Always run Streamlit from the **project root**, otherwise relative paths to the `.pkl` and `.csv` files may fail.

---

## Project Structure

At a high level:

- **`app.py`**: Main Streamlit application that orchestrates face, voice, and recommendation models.
- **`Image_processing/`**:
  - `Image_feature_training.ipynb` – image feature extraction and model training.
  - `image_model.pkl`, `image_label_encoder.pkl` – trained face model and encoder.
  - `images/` – raw images used during training.
  - `image_dataset/` – structured dataset for train/test splits.
- **`audio_processing/`**:
  - Audio feature extraction and training notebooks.
  - `audio_model.pkl`, `audio_label_encoder.pkl` – trained voice model and encoder.
  - `raw_data/` – original audio samples.
- **`prediction_functionality/`**:
  - `customer_social_profiles.csv`, `customer_transactions.csv` – tabular source data.
  - `merged_customer_data.csv` – engineered dataset used by the app.
  - `product_recommendation_model.pkl`, `product_category_encoder.pkl` – recommendation artifacts.
  - Notebook(s) for model training and evaluation.
- **`venv/`**: Local virtual environment (not required if you prefer your own env, but included here).
- **`requirements.txt`**: Python package dependencies.

---

## Tech Stack

- **Language**
  - Python (3.10+)

- **Web Framework**
  - Streamlit – fast prototyping of data apps and ML demos.

- **Machine Learning**
  - scikit‑learn – `RandomForestClassifier`, `LogisticRegression`, `train_test_split`, `classification_report`, etc.

- **Image Processing**
  - Pillow (PIL) – image loading and resizing.
  - TensorFlow/Keras `ImageDataGenerator` – data augmentation in notebooks.

- **Audio Processing**
  - librosa – MFCC extraction and audio augmentations.

- **Data & Analysis**
  - pandas, numpy – data manipulation and feature engineering.
  - matplotlib, seaborn – analysis and visualization in notebooks.

---

## Troubleshooting

- **`streamlit` not recognized (Windows)**  
  - Ensure the virtual environment is activated:
    - `.\venv\Scripts\activate`
  - Then run:
    - `python -m streamlit run app.py`

- **`FileNotFoundError` for `.pkl` or `.csv`**
  - Confirm you are in the **project root** when running Streamlit.
  - Verify that the files exist at:
    - `Image_processing/image_model.pkl`
    - `Image_processing/image_label_encoder.pkl`
    - `audio_processing/audio_model.pkl`
    - `audio_processing/audio_label_encoder.pkl`
    - `prediction_functionality/product_recommendation_model.pkl`
    - `prediction_functionality/merged_customer_data.csv`

- **Version / dependency issues**
  - Recreate the environment from scratch:

    ```bash
    rm -rf venv  # or delete the folder on Windows
    python -m venv venv
    .\venv\Scripts\activate  # or source venv/bin/activate
    pip install -r requirements.txt
    ```

---

## Team & Contributions

- **Uwingabire Caline – Recommender Lead**
  - Product recommendation model.
  - Tabular EDA and data cleaning.

- **Peace Keza – Computer Vision Lead**
  - Facial recognition model.
  - Image collection and preprocessing.

- **Theodora Egbunike – Audio Lead & Report Compiler**
  - Voiceprint model.
  - Audio collection, augmentation, and processing.
  - Compilation and editing of the final project report.

- **Senga Kabare – Integration Lead**
  - System integration.
  - `app.py` Streamlit application.
  - Final demo video and end‑to‑end wiring of all components.

---

If you have questions or would like to extend this project (e.g., add more users, improve the recommendation model, or integrate additional biometric modalities), feel free to open an issue or fork the repository.


