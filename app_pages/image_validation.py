import streamlit as st
from PIL import Image
import pandas as pd

from utils import data_utils

def main():

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)
    
    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.markdown("""<h3 style='text-align: left; color: black;'>
                User Information</h4>""",
                unsafe_allow_html=True)
    
    left_1, right_1 = st.columns(2)
    with left_1:
        user_name = st.text_input(label = "Name:")
    with right_1:
        user_email = st.text_input(label = "Email:")

    left_2, right_2 = st.columns(2)
    with left_2:
        user_lab = st.text_input(label = "Lab:")
    with right_2:
        user_date = st.date_input(label = "Date:")

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Goal</h4>""",
                unsafe_allow_html=True)
    
    user_goal = st.radio(label='For what purpose are the images being labeled?',
                    options=['Performance Verification', 'Model Retraining'],
                    index=None)
    
    user_selec = st.radio(label='What selection method would you like to use?',
                          options=('entropy_score',
                                   'least_confident_score',
                                   'least_margin_score'),
                          format_func={'entropy_score': 'Entropy',
                                       'least_confident_score': 'Least Confidence',
                                       'least_margin_score': 'Least Margin'}.__getitem__,
                          index=None)
    
    user_number = st.selectbox(label='What is the prefered image batch size?',
                               options=list(range(10,100,10)),
                               index=None)

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Image Validation</h4>""",
                unsafe_allow_html=True)
    
    if user_number!=None and user_selec!=None and user_goal!=None:
        metrics_df = pd.read_parquet('data/inventory_df_with_scores.parquet.gzip')
        metrics_df['blob_name'] = metrics_df['image_path'].apply(lambda x: x.split('NAAMES/')[1])
        
        if user_goal == 'Model Retraining':
            filtered_df = metrics_df.sort_values(user_selec, ascending=True).head(user_number)
        elif user_goal == 'Performance Verification':
            filtered_df = metrics_df.sample(n=user_number)

        try:
            if 'image_iterator' not in st.session_state:
                st.session_state['image_iterator'] = data_utils.get_images(filtered_df, 1)
        except:
            st.error("Failed to instantiate class")

    image_set = []
    selection = {'entropy_score': 'Entropy',
                 'least_confident_score': 'Least Confidence',
                 'least_margin_score': 'Least Margin'}

    if all((user_name!="", user_email!="", user_lab!="", user_goal is not None, user_selec is not None, user_number is not None)):
    
        with st.form("plankton_form", clear_on_submit=True):   
            try:
                if 'image_iterator' in st.session_state:
                    image_set = next(st.session_state['image_iterator'])
            except:
                st.error("Failed to load next image set")

            for image in image_set: 

                st.image(image[0])
                ml_label = str(image[1][3])
            
                st.write("ML Generated Label: " + ml_label)
                st.write(selection[user_selec], " Score: ", image[1][user_selec])

                user_label = st.selectbox(
                    label = "Select the correct phytoplankton subcategory:",
                    options = ['Cilliate',
                                'Chloro',
                                'Crypto',
                                'Diatom',
                                'Dictyo',
                                'Dinoflagellate',
                                'Eugleno',
                                'Prymnesio',
                                'Other',
                                'Not phytoplanton'],
                    index=None
                )

            submitted = st.form_submit_button("Submit")
            if submitted:
                # Will this data be sent to a SQL Database?????????
                new_data = [image[1]['index'],
                        image[1]['image_path'],
                        image[1]['pred_label'],
                        image[1]['pred_class'],
                        image[1]['entropy_score'],
                        image[1]['least_confident_score'],
                        image[1]['least_margin_score'],
                        user_label,
                        user_name,
                        user_email,
                        user_lab,
                        user_date,
                        user_goal,
                        user_selec]
                st.write("The previous image was labeled: ", user_label)

