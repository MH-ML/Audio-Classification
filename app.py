import pickle

import librosa
import numpy as np
import requests
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Audio Classification", page_icon=":speaker:", layout="centered")

# Custom CSS
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Function to load SVM model and scaler from GitHub
@st.cache_resource
def load_model_and_scaler(model_url, scaler_url):
    model_response = requests.get(model_url)
    scaler_response = requests.get(scaler_url)
    model_content = model_response.content
    scaler_content = scaler_response.content
    svm_model = pickle.loads(model_content)
    scaler = pickle.loads(scaler_content)
    return svm_model, scaler

# Load SVM model and scaler from GitHub
svm_model_url = 'https://github.com/MH-ML/Audio-Classification/raw/main/svm_model.pkl'
scaler_url = 'https://github.com/MH-ML/Audio-Classification/raw/main/scaler.pkl'
svm_model, scaler = load_model_and_scaler(svm_model_url, scaler_url)

# Function to extract MFCC features
@st.cache_data
def extract_feature(file_path, n_mfcc=40, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

# Function to predict audio
@st.cache_data
def predict_audio(audio_file):
    try:
        # Save the uploaded file to a temporary location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_file_path = tmp.name
            tmp.write(audio_file.read())

        # Extract MFCC features
        mfcc = extract_feature(temp_file_path)
        mfcc_scaled = scaler.transform([mfcc])

        # Make predictions using the SVM model
        prediction = svm_model.predict(mfcc_scaled)

        # Return the predicted label
        return 'Break' if prediction[0] == 1 else 'Non Break'
    except Exception as e:
        return str(e)

# Main Streamlit app
def main():
    st.title(":speaker: Machine Tool Break or Non Break Classification")

    uploaded_file = st.file_uploader("Upload an audio file (MP3 or WAV format)", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')

        if st.button("Predict"):
            with st.spinner('Predicting...'):
                prediction = predict_audio(uploaded_file)
                if prediction.startswith("LibsndfileError"):
                    st.error("Error: Unsupported audio format or corrupted file.")
                else:
                    st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()