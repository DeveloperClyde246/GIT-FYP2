import streamlit as st
import tempfile
import numpy as np
import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
from tone_analysis_dashboard.preprocess_function import *

# Streamlit app
st.title("Emotion Analysis")


col1, col2 = st.columns([2, 5])  # Left column is wider than the right column

# Video directory
video_dir = "uploaded_videos"
uploaded_file = None

with col1:
    st.subheader("Video")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.write("Question: Apa yang anda ketahui tentang Ionic?")
    # Loop through video files in the directory
    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_file = video_path
            st.video(uploaded_file)
            #st.success(f"Video {video_filename} loaded successfully!")

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(uploaded_file, 'rb') as video_file:
            tfile.write(video_file.read())

with col2:
    # Create tabs for Personality Traits and Emotions Analysis
    tab1, tab2 = st.tabs(["Extracted Audio Features", "Emotions Analysis"])
    with tab1:
            # Show a processing message
        with st.spinner('Processing...'):
            # Extract audio from video
            audiofile = extract_audio(tfile.name)

            # Preprocess audio
            features = preprocess_audio(audiofile)

            # Load the label encoder and feature scaler
            emotion_le = joblib.load('emotion_model/emotion_label_encoder.joblib')
            emotion_scaler = joblib.load('emotion_model/emotion_feature_scaler.joblib')
            
            # Predict emotions
            emotion_results = predict_emotion(features, emotion_scaler, emotion_le)



            # Display the extracted audio features
            st.write("Extracted Audio Features: ")
            st.write(features)
            st.write("Shape of the features:", features.shape)

            col1, col2 = st.columns([1, 1]) 
            with col1:
                # Visualize the audio waveform
                st.write("Audio Waveform:")
                y, sr = librosa.load(audiofile, sr=None)
                fig, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set(title='Waveform of the Audio')
                st.pyplot(fig)

            with col2:
                # Visualize the spectrogram
                st.write("Spectrogram:")
                fig, ax = plt.subplots()
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')
                st.pyplot(fig)

    with tab2: 
        # Display Emotions results
        #st.header("Result")
        for model, scores in emotion_results.items():
            st.subheader(model)
            if len(emotion_le.classes_) == len(scores):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    for emotion, score in zip(emotion_le.classes_, scores):
                        st.write(f"{emotion}: {score * 100:.2f}%")
                with col2:
                    # Plot pie chart with consistent colors
                    fig = px.pie(values=[s * 100 for s in scores], names=emotion_le.classes_, title=f"{model} Emotions",
                                    color=emotion_le.classes_, color_discrete_sequence=px.colors.qualitative.Plotly)
                    st.plotly_chart(fig)

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_emotion}.")

                # Display confidence and consistency metrics
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
            else:
                st.error("Mismatch between emotions and scores. Check model output!")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")