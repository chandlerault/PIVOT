"""
Code that executes the contents of the About page
and is called by the main app.py script.

Functions:
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import streamlit as st

def main():
    """
    Executes the Streamlit formatted HTML displayed on the About subpage.
    The images shown here describe the purpose and information flow of the project.
    """
    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            About<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    # Display markdown for the purpose of the project
    st.markdown("""<h3 style='text-align: left; color: black;'>
                Project Goal</h3>""",
                unsafe_allow_html=True)

    st.write("""Phytoplankton play a key role in supporting all life on Earth. Given the
             quantity and size of phytoplankton cells, there is a need to accurately and
             efficiently quantify them to evaluate their impact in the global Carbon Cycle.
             The current approach of label validation in this field is to manually label
             each image. This approach is complex, inefficient, time-consuming, and does not
             track overall performance metrics or confidence. A streamlined, efficient, in-
             house approach is needed to validate machine learning (ML) model-labeled images,
             display accuracy statistics, and allow researchers to correct incorrectly
             labeled images.""")

    st.write("""Researchers in the Air-Sea Interaction & Remote Sensing department at the
             Applied Physics Lab at the University of Washington have partnered with a
             Masters of Science in Data Science capstone group to create an interactive
             tool for validating Convolution Neural Network (CNN) classified phytoplankton
             images. This tool allows researchers to confirm and correct CNN-labeled images,
             while displaying relevant accuracy and other useful statistics.""")

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Data Pipeline</h3>""",
                unsafe_allow_html=True)

    # Center system diagram image and display system diagram description

    st.markdown("""<h4 style='text-align: center; color: black;'>
                System Diagram</h4>""",
                unsafe_allow_html=True)

    left_1, middle_1, right_1 = st.columns([2,3,2])
    with middle_1:
        st.image("images/SYS.svg")
    with left_1 and right_1:
        pass

    st.write("""The PIVOT App works as a user interface for researcher to label and review
             summary statistics for CNN-labeled image validation. The Image Validation and
             Summary statistics pages can be reached on the navigation sidebar. The information
             used to populate the *User Information* and *Session Specifications* sections,
             images requiring validation, and summary metrics are all stored in a SQL database
             hosted on
             [Microsoft Azure](https://azure.microsoft.com/en-us/get-started/azure-portal)
             . In order to modify, add, or change attributes, you must obtain access to
             appropriate Azure subscription and Virtual Machine were metrics are calculated.
             All images outputted by the Image FlowCytobot are accessed from [Azure Blob
             Storage](https://azure.microsoft.com/en-us/products/storage/blobs/?ef_id=_k_CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE_k_&gad_source=1&gclid=CjwKCAiAlJKuBhAdEiwAnZb7lU3UHKC7kxkUA8gz1C1HdeqXUScz1WVwDxDPZGeUlTMrWUpk8isslBoC9H0QAvD_BwE)
             . File paths to each image are collected from Blog Storage and saved in the
             SQL Database. The model's performance and prediction metrics are calculated
             and served on
             [Azure Machine Learning (ML)](https://azure.microsoft.com/en-us/products/machine-learning)
             . These metrics are saved to the SQL Database as well.""")

    # Center system diagram image and display image validation flow diagram description

    st.markdown("""<h4 style='text-align: center; color: black;'>
                Image Validation Flow Diagram</h4>""",
                unsafe_allow_html=True)

    left_2, middle_2, right_2 = st.columns([1,5,1])
    with middle_2:
        st.image("images/IV.svg")
    with left_2 and right_2:
        pass

    st.write("""The Image Validation Flow Diagram visualizes the sequence of events when
             using the image validation tool. Items in blue represent tasks completed by
             the user through the PIVOT app. These user entries and outputs are fed either
             into the SQL Database directly, used in stored procedures, or connect to the
             Azure ML client. First, a user will input a pretrained data into Azure ML
             before being able to access the information on the app. They can train a new 
             models and extract relevant statistics through our data flow, ultimately serving
             them in Azure ML. Once the desired models are inputted and served, the user will
             fill out several forms on the app including the Email, Session, and Label forms.
             The information gathered in these forms feed directly to tables in the SQL
             Database.""")

    # Center system diagram image and display ER diagram description

    st.markdown("""<h4 style='text-align: center; color: black;'>
                Entity Relationship (ER) Diagram</h4>""",
                unsafe_allow_html=True)

    left_3, middle_3, right_3 = st.columns([1,6,1])
    with middle_3:
        st.image("images/ER.svg")
    with left_3 and right_3:
        pass

    st.write("""The diagram above visualizes the relationships between all tables in
             the SQL Database. For a full description of all variables saved to the
             database, please refer to the GitHub README for this project.""")
