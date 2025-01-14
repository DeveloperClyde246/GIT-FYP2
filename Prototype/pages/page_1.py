import streamlit as st

# Function to switch pages
def switch_page(page_name):
    st.session_state.current_page = page_name

# Page 1 content
st.title("Page 1")
st.write("This is the content of Page 1.")

# Button to return to the home page
if st.button("Return to Home"):
    st.switch_page("main.py")
