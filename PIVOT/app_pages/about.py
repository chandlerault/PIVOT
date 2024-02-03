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
                Project Goal</h4>""",
                unsafe_allow_html=True)
    
    st.write("""Phytoplankton play a key role in supporting all life on Earth. Given the
             quantity and size of phytoplankton cells, there is a need to accuratly and
             efficently quanitfy them to evaluate their impact in the global Carbon Cycle.
             The current approach of label validation in this field is to manually label 
             each image. This approach is complex, inefficent, time-consuming, and does not
             track overall perfomance metrics or confidence. A streamlined, efficent, in-
             house approach is needed to validate machine learning (ML) model-labeled images,
             display accuracy statistics, and allow researchers to correct incorrectly
             labeled images.""")
    
    st.write("""Researchers in the Air-Sea Interaction & Remote Sensing department at the
             Applied Physics Lab at the University of Washington have partnered with a
             Masters of Science in Data Science capstone group to create an interactive
             tool for validating Convolution Neural Network (CNN) classified phytoplankton
             images. This tool allows researchers to confirm and correct CNN-labeled images,
             while displaying relavent accuracy and other useful statistics.""")
    
    st.markdown("""<h3 style='text-align: left; color: black;'>
                Data Pipeline</h4>""",
                unsafe_allow_html=True)