"""
Code that executes the contents of the Resources page
and is called by the main app.py script.

Functions:
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import streamlit as st

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Resources subpage. This
    page contains useful links and information for users labeling phytoplankton images.
    """

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Phytoplankton Classification Resources</h4>""",
                unsafe_allow_html=True)
