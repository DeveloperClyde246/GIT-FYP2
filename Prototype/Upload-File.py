import streamlit as st
import os

# Create a directory to save uploaded videos (if it doesn't exist)
video_dir = "uploaded_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Function to clear the uploaded_videos folder
def clear_previous_videos():
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Set the page title
st.title("Upload Your Video")

# File uploader for video
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Clear the previous videos before saving the new one
    clear_previous_videos()

    # Save the uploaded video to the `uploaded_videos` folder
    temp_file_path = os.path.join(video_dir, uploaded_video.name)

    # Write the uploaded video directly to the target directory
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(temp_file_path)  # Display the uploaded video
    st.success(f"Video saved successfully at {temp_file_path}!")

    if st.button("Proceed"):
            st.switch_page("pages/home.py")