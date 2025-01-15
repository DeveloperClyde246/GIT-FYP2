import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
import cv2
import time

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}<style>',unsafe_allow_html=True)

st.set_page_config(layout="wide")

# Set the page title
st.title("Interview Analysis")

# # Custom CSS to inject
# st.markdown("""
# <style>
# .streamlit-container {
#     border: 2px solid #111;
#     padding: 10px;
# }
# </style>
# """, unsafe_allow_html=True)

# Create two columns with custom width ratios
col1, col2 = st.columns([3, 1])  # Left column is wider than the right column

with col1:
    st.header("Video and Details")
    # Define the video directory
    video_dir = "uploaded_videos"

    # Check if there are any uploaded videos
    if os.listdir(video_dir):
        # Show all uploaded videos
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            st.video(video_path)

            st.subheader("Transcription")
            st.text("Ionic adalah framework yang membangun aplikasi mobile dengan menggunakan html css dan javascript")
    else:
        st.warning("No videos uploaded yet.")

    if st.button("Choose Other Video"):
            st.switch_page("main.py")
    # # Video upload functionality
    # uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    # if uploaded_video is not None:
    #     st.video(uploaded_video)
    #     st.success("Video uploaded successfully!")

    #     # Save the uploaded video to a temporary file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
    #         temp_file.write(uploaded_video.read())
    #         temp_video_path = temp_file.name

    #     # Start timing the processing
    #     start_time = time.time()

    #     # End timing the processing
    #     end_time = time.time()
    #     total_time = end_time - start_time

    #     # Display the total processing time
    #     st.write(f"Total processing time: {total_time:.2f} seconds")

    #     # Clean up the temporary file
    #     os.remove(temp_video_path)

with col2: 
    st.header("Analysis")
    if st.button("Home"):
        st.switch_page("main.py")
    if st.button("Page 1"):
        st.switch_page("pages/page_1.py")
    if st.button("Page 2"):
        st.switch_page("pages/page_2.py")
