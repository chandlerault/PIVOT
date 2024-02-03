import streamlit as st
from PIL import Image
import pandas as pd

def main():

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)
    
    st.markdown("""<h1></h1>""", unsafe_allow_html=True)
    
    st.markdown("""<h3 style='text-align: left; color: black;'>
                Phytoplankton Classification Resources</h4>""",
                unsafe_allow_html=True)