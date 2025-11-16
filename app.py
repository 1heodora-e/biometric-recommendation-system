import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import joblib
import os


# Load pre-trained models
image_model = joblib.load('image/image_model.pkl')
image_le = joblib.load('image/image_label_encoder.pkl')

audio_model = joblib.load('voice/audio_model.pkl')
audio_le = joblib.load('voice/audio_label_encoder.pkl')

product_model = joblib.load('predictor/product_recommendation_model.pkl')


# Load merged customer data
merged_file = 'predictor/merged_customer_data.csv'
if not os.path.exists(merged_file):
    st.error(f"Error: {merged_file} not found. Please make sure the file exists.")
    st.stop()

merged_data = pd.read_csv(merged_file)


# Define feature columns
feature_columns = [
    'engagement_score', 
    'purchase_interest_score', 
    'review_sentiment_encoded',
    'avg_purchase_amount', 
    'total_spent', 
    'purchase_count', 
    'avg_rating'
]

for col in feature_columns:
    if col not in merged_data.columns:
        merged_data[col] = 0

product_categories = ['clothes', 'sports', 'electronic', 'books', 'groceries']



# Helper Functions

def predict_image_class(image_file, threshold=0.6):
    try:
        img = Image.open(image_file).resize((64, 64))
        img_array = np.array(img).flatten().reshape(1, -1)
        probs = image_model.predict_proba(img_array)[0]
        max_prob = np.max(probs)
        label = image_le.inverse_transform([np.argmax(probs)])[0]
        return label if max_prob >= threshold else "unauthorized"
    except Exception as e:
        return f"Error: {e}"

def predict_voice(audio_file, threshold=0.55):
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=42)
        features = np.mean(mfccs.T, axis=0).reshape(1, -1)
        probs = audio_model.predict_proba(features)[0]
        max_prob = np.max(probs)
        label = audio_le.inverse_transform([np.argmax(probs)])[0]
        return label if max_prob >= threshold else "unauthorized"
    except Exception as e:
        return f"Error: {e}"

def recommend_all_customers():
    """Runs the product model AFTER both verifications succeed."""
    results = []

    for _, row in merged_data.iterrows():
        features = row[feature_columns].values.reshape(1, -1)

        try:
            probs = product_model.predict_proba(features)[0]
        except:
            probs = np.full(len(product_categories), 1/len(product_categories))

        top_idx = np.argsort(probs)[::-1][:3]

        for i in top_idx:
            results.append({
                "customer_id": row["customer_id"],
                "Category": product_categories[i],
                "Probability": f"{probs[i]:.2%}"
            })
    return pd.DataFrame(results)



# Streamlit UI (NEW FLOW)


st.title("🔐 Face → Parameter → Voice Verification → Recommendation")
st.write("Authenticate with your **face first**, enter ANY parameter, then verify with **voice** to unlock recommendations.")

# STEP 1 — FACE VERIFICATION
face_file = st.file_uploader("Step 1: Upload your face image", type=["jpg", "png", "jpeg"])

if face_file:
    st.info("Verifying face...")
    face_result = predict_image_class(face_file)
    st.write(f"Face result: **{face_result}**")

    if face_result == "unauthorized":
        st.error("Face verification failed.")
        st.stop()

    st.success("Face verified! Proceed to parameter input.")

    # STEP 2 — PARAMETER OF ANY TYPE
    user_param = st.text_input("Step 2: Enter ANY parameter (string, number, etc.)")

    if user_param:
        st.info(f"Parameter received: **{user_param}**")

        # STEP 3 — VOICE VERIFICATION
        audio_file = st.file_uploader("Step 3: Upload your voice sample", type=["wav", "mp3", "flac"])

        if audio_file:
            st.info("Verifying voice...")
            voice_result = predict_voice(audio_file)
            st.write(f"Voice result: **{voice_result}**")

            if voice_result == "unauthorized":
                st.error("Voice verification failed.")
                st.stop()

            st.success("Voice verified! 🎉")

            # STEP 4 — RUN PRODUCT PREDICTION MODEL
            st.info("Running product recommendation model...")
            df = recommend_all_customers()

            st.write("### 🔥 Recommendations:")
            st.dataframe(df)
