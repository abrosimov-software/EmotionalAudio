import streamlit as st
import requests
import os

# Set FastAPI URL from environment variable
fastapi_url = os.getenv("FASTAPI_URL", "http://model_api:8000")

st.title("ML Model Audio Prediction Interface")

# Retrieve available models from the API
try:
    response = requests.get(f"{fastapi_url}/models")
    response.raise_for_status()
    models = response.json().get("models", [])
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching models: {e}")
    models = []

# Check if models are available
if models:
    # Allow the user to select a model
    model_name = st.selectbox("Select a model", models)

    # Allow the user to upload an audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    # Check if a file was uploaded
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")  # Playback the audio

        # When the user clicks the predict button
        if st.button("Predict"):
            # Send the audio file and model name to FastAPI
            try:
                files = {'file': (audio_file.name, audio_file, audio_file.type)}
                params = {'model_name': model_name}
                response = requests.post(f"{fastapi_url}/predict", params=params, files=files)
                response.raise_for_status()

                # Get the prediction result
                prediction = response.json().get("prediction", "")
                st.success(f"Prediction: {prediction}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
else:
    st.error("No models available.")