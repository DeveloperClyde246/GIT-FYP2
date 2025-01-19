import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from Preprocessor import Preprocessor
import tempfile
import os
import numpy as np
import cv2
import time
from facial_expression_recognition.Preprocessor import Preprocessor

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('Model/model3.h5')#load model

st.set_page_config(layout="wide")
st.title("Facial Expression Analysis")

col1, col2 = st.columns([2, 5])  

with col1:
    st.header("Video")

    st.write(" ")
    st.write(" ")
    st.write(" ")

    video_dir = "uploaded_videos"
    uploaded_video = None

    st.write("Question: Apa yang anda ketahui tentang Ionic?")

    #video loading
    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_video = video_path
            st.video(uploaded_video)
            #st.success(f"Video {video_filename} loaded successfully!")

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            with open(uploaded_video, 'rb') as video_file:
                temp_file.write(video_file.read())
            temp_video_path = temp_file.name


        #time calculation
        start_time = time.time()

        with st.spinner('Processing...'):
            preprocessor = Preprocessor()
            preprocessed_data = preprocessor.preprocess(temp_video_path)
            st.write(f"Extracted {len(preprocessed_data)} frames from the video.")

            processed_frames = np.array(preprocessed_data)
            predictions = model.predict(processed_frames)
            predicted_emotions = np.argmax(predictions, axis=1)

            #end timing the processing
            end_time = time.time()
            total_time = end_time - start_time

            #map predictions to emotion labels
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
            emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

            #total processing time
            st.write(f"Total processing time: {total_time:.2f} seconds")

            #Clean up the temporary file
            os.remove(temp_video_path)

with col2:
    if uploaded_video is not None and len(predicted_emotions) > 0:
        #pie chart based on the emotion counts
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90, colors=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
        ax.axis('equal')  #equal aspect ratio ensures that pie is drawn as a circle

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Facial Expression Distribution")
            st.pyplot(fig) #display the pie chart

        with col2:
            st.write("###  ")
            st.write(" ")
            st.write(" ")
            #rename the dataframe columns
            emotion_counts = emotion_counts.reset_index()
            emotion_counts.columns = ['Emotions', 'Frames']
            st.table(emotion_counts)
        #display the message for the maximum emotion
        max_emotion = emotion_counts.loc[emotion_counts['Frames'].idxmax()]['Emotions']
        st.write(f"Final result: The facial expression of the candidate is {max_emotion} in this video.")
    else:
        st.write("Upload a video to view the emotion distribution.")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")