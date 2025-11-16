A Multimodal Biometric Authentication System for Personalized Product Recommendation

This repository contains the source code for our Machine learning pipeline Formative 2 project demonstrating a secure, end-to-end user authentication flow. The system is built as an interactive Streamlit web application that uses a multimodal biometric approach, requiring users to pass both facial recognition and voiceprint verification before granting access to a personalized product recommendation engine.

This project successfully integrates three distinct machine learning models into a single, functional web application.

System Demonstration
A brief video demonstrating the full, successful transaction flow as well as a failed "unauthorized user" attempt.

https://drive.google.com/file/d/1F-BTXwHLijkuj9gXApjubqmmkFqvrHHs/view?usp=sharing


📖 Table of Contents
Key Features

How It Works: The System Flow

Installation & Usage

Project Structure

Tech Stack

Team & Contributions

Key Features
This project is built on three distinct, independently trained machine learning models:

Product Recommendation Model:

A RandomForestClassifier trained on a merged dataset of customer social profiles and past transactions to predict a user's likely next purchase.

Final Model Accuracy: 23.08% (Identified as a key area for improvement, likely due to small dataset size and weak feature correlation).

Facial Recognition Model:

A LogisticRegression model trained on a custom-collected dataset of 4 users, each with 3 facial expressions (neutral, smile, surprised) and augmentations.

Voiceprint Verification Model:

A RandomForestClassifier trained on a custom-collected audio dataset. To ensure robustness, the 8 original samples were expanded to 56 samples using 6 data augmentation techniques (noise, pitch shift, time stretch, etc.).

Final Model Accuracy: 91.67%

How It Works: The System Flow
The final Streamlit application enforces a strict, multimodal security logic to protect the user's personalized data.

The user opens the web app and is prompted to upload a face image.

The system runs the Facial Recognition Model. If the face is not recognized, Access is Denied.

If the face is recognized (e.g., as "Theodora"), the app prompts the user to upload their voice sample.

The system runs the Voiceprint Verification Model.

The system performs a biometric match: if (face_prediction == "Theodora") AND (voice_prediction == "Theodora").

If the check fails (e.g., voice is "Keza"), Access is Denied.

If both biometrics match, the user is authenticated. The system then runs the Product Recommendation Model and displays the personalized result on the web page.

Installation & Usage
Follow these steps to set up the virtual environment and run the application.

1. Prerequisites
Python 3.9 or higher

git

2. Installation
Clone the repository:

Bash

git clone [https://github.com/1heodora-e/biometric-recommendation-system.git]
Navigate to the project directory:

Bash

cd [biometric-recommendation-system]
Create a virtual environment:

Bash

python -m venv venv
Activate the virtual environment:

On Windows (PowerShell/CMD):

Bash

.\venv\Scripts\activate
On macOS/Linux:

Bash

source venv/bin/activate
Install the required packages: (This project includes a requirements.txt file for easy setup)

Bash

pip install -r requirements.txt
3. Running the Streamlit App
With your virtual environment active, run the main Streamlit application:

Bash

streamlit run app.py



Tech Stack
Python 3.10

Web Framework: Streamlit

Machine Learning: Scikit-learn (for RandomForest, LogisticRegression, train_test_split, classification_report)

Audio Processing: Librosa (for MFCCs, augmentations)

Image Processing: OpenCV / PIL

Data Manipulation: Pandas & Numpy

Analysis: Jupyter Notebook

Plotting: Matplotlib & Seaborn

Team & Contributions
This project was completed by a team of four members, with roles as defined in our final report.

Uwingabire Caline (Recommender Lead):

Recommender Model, Tabular EDA & Cleaning.

Peace Keza (CV Lead):

Facial Recognition Model, Image Collection & Processing.

Theodora Egbunike (Audio Lead & Report Compiler):

Voiceprint Model, Audio Collection, Augmentation, & Processing.

Compiled and edited the final project report.

Senga Kabare (Integration Lead):

System Integration, app.py Script & Final Demo Video.