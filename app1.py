import pickle

import librosa
import numpy as np
import streamlit as st

# Load the trained SVM model and scaler
svm_model = pickle.load(open('F:\\class\\DeepLearning\\Classification audio\\svm_model.pkl', 'rb'))
scaler = pickle.load(open('F:\\class\\DeepLearning\\Classification audio\\scaler.pkl', 'rb'))

# Function to extract MFCC features
def extract_feature(file_path, n_mfcc=40, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

# Streamlit app
def main():
    st.title("Machine Tool Break or Non Break Classification")

    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Save the uploaded file to a temporary location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_file_path = tmp.name
            tmp.write(audio_file.read())

        # Extract MFCC features
        mfcc = extract_feature(temp_file_path)
        mfcc_scaled = scaler.transform([mfcc])

        # Add a "Predict" button
        if st.button("Predict"):
            # Make predictions
            prediction = svm_model.predict(mfcc_scaled)

            # Display the prediction
            if prediction[0] == 1:
                st.success("Prediction: Break")
            else:
                st.success("Prediction: Non Break")

if __name__ == "__main__":
    main()