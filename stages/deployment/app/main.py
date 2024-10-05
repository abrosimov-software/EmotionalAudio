import streamlit as st
import requests
import os

# Set FastAPI URL from environment variable
fastapi_url = os.getenv("FASTAPI_URL", "http://api:8000")

st.title("ML Model Audio Prediction Interface")

# Allow the user to upload an audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Check if a file was uploaded
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")  # Optionally, playback the audio for confirmation

    # When the user clicks the predict button
    if st.button("Predict"):
        # Send the audio file to FastAPI as multipart form data
        try:
            files = {'file': (audio_file.name, audio_file, audio_file.type)}
            response = requests.post(f"{fastapi_url}/predict", files=files)
            response.raise_for_status()  # Check for errors

            # Get the prediction result
            prediction = response.json().get("prediction", [])
            st.success(f"Prediction: {prediction}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
