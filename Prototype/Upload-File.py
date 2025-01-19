import streamlit as st
import os

#function: create folder to save uploaded videos
video_dir = "uploaded_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

#function: clear the folder
def clear_previous_videos():
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

#UI
st.title("Upload Your Video")
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    clear_previous_videos() #clear vid uploaded before
    temp_file_path = os.path.join(video_dir, uploaded_video.name) #save vid to folder

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.read())  # Write the uploaded video 

    st.video(temp_file_path)  
    st.success(f"Video saved successfully at {temp_file_path}!")

    if st.button("Proceed"):
            st.switch_page("pages/1_Home.py")