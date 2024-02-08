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
                Project Goal</h3>""",
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
                Data Pipeline</h3>""",
                unsafe_allow_html=True)
    
    left_1, middle_1, right_1 = st.columns([2,3,2])
    with middle_1:
        st.image("images/SYS.png")

    st.markdown("""<h4 style='text-align: center; color: black;'>
                System Diagram</h4>""",
                unsafe_allow_html=True)

    st.write("""The PIVOT App works as a user interface for researcher to label and review
             summary statistics for CNN-labeled image validation. The Image Validation and
             Summary statistics pages can be reached on the navigation sidebar. The information
             used to populate the *User Information* and *Session Specifications* sections,
             images requiring validation, and summary metrics are all stored in a SQL database
             hosted on
             [Microsoft Azure](https://azure.microsoft.com/en-us/get-started/azure-portal)
             . In order to modify, add, or change attributes, you must obtain access to
             appropirate Azure subscription and Virtual Machine were metrics are calculated.
             All images outputted by the Image FlowCytobot are accessed from [Azure Blob
             Storage](https://azure.microsoft.com/en-us/products/storage/blobs/?ef_id=_k_CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE_k_&gad_source=1&gclid=CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE)
             . File paths to each image are collected from Blog Storage and saved in the 
             SQL Database. The model's performance and prediction metrics are calculated
             and served on
             [Azure Machine Learning (ML)](https://azure.microsoft.com/en-us/products/machine-learning)
             . These metrics are saved to the SQL Database as well. For a full description on
             the variables saved to the SQL Database, please refer to the GitHub README for
             this project.""")

    st.image("images/ER.png")

    st.markdown("""<h4 style='text-align: center; color: black;'>
                Entity Relationship (ER) Diagram</h4>""",
                unsafe_allow_html=True)

    st.image("images/IV.png")

    st.markdown("""<h4 style='text-align: center; color: black;'>
                Image Validation Flow Diagram</h4>""",
                unsafe_allow_html=True)
