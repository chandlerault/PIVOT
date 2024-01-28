import streamlit as st
from PIL import Image
import pandas as pd

from utils import data_utils

def get_user_experience(num_labels, domain):
    if domain == "No" and num_labels == "None":
        return 1
    elif domain == "No" and num_labels == "25 to 100":
        return 2
    elif domain == "No" and num_labels == "100 to 500":
        return 3
    elif domain == "No" and num_labels == "500 to 1000":
        return 4
    elif domain == "No" and num_labels == "1000+":
        return 5
    elif domain == "Yes" and num_labels == "None":
        return 2
    elif domain == "Yes" and num_labels == "25 to 100":
        return 3
    elif domain == "Yes" and num_labels == "100 to 500":
        return 4
    elif domain == "Yes" and num_labels == "500 to 1000":
        return 5
    elif domain == "Yes" and num_labels == "1000+":
        return 5

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
    
    #############################################################
    #           FAKE VARIABLES ARE BEING USED HERE
    #
    #   + dummy_email
    #
    #############################################################

    dummy_email = {'u_id': 0,
                   'email': 'yamina.katariya@gmail.com',
                   'name': 'Yamina Katariya',
                   'experience': 1,
                   'lab': 'UW Capstone'}
    
    left_1, right_1 = st.columns(2)
    with left_1:
        user_email = st.text_input(label = "Please enter your email:")

    # Retreive username and lab from database
    if user_email is not "" and user_email != dummy_email['email']:
        left_2, right_2 = st.columns(2)
        with left_2:
            user_name = st.text_input(label = "Name:")
        with right_2:
            user_lab = st.text_input(label = "Lab:")
            
        left_3, right_3 = st.columns(2)
        with left_3:
            user_domain = st.radio(label="Do you have experience in this field?",
                                   options=['Yes', 'No'],
                                   index=None)
        with right_3:
            user_num_labels = st.radio(label="Approximately how many images have you labeled?",
                                       options=['None', '25 to 100',
                                                '100 to 500', '500 to 1000',
                                                '1000+'],
                                        index=None)
            
        user_experience = get_user_experience(user_num_labels, user_domain)
    elif user_email == dummy_email['email']:
        st.markdown("""<h5 style='text-align: left; color: black;'>
                User Found</h5>""",
                unsafe_allow_html=True)
        st.write("Name: " + dummy_email['name'])
        st.write("Experience: " + str(dummy_email['experience']))
        st.write("Lab: " + dummy_email['lab'])        
        st.write("Email: " + dummy_email['email'])

        user_name = dummy_email['name']
        user_lab = dummy_email['lab']
        user_experience = dummy_email['experience']

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Session Specifications</h4>""",
                unsafe_allow_html=True)
    
    #############################################################
    #           FAKE VARIABLES ARE BEING USED HERE
    #
    #   + session_model --> options
    #
    #############################################################

    left_4, right_4 = st.columns(2)
    with left_4:
        session_model = st.selectbox(label='Select the model you wish to validate:',
                                    options=('xxx', 'yyy', 'zzz'),
                                    format_func={'xxx': 'CNN ver. 1',
                                                'yyy': 'CNN ver. 2',
                                                'zzz': 'CNN ver. 3'}.__getitem__,
                                    index=None)
    with right_4:
        session_dissim = st.selectbox(label='What selection method would you like to use?',
                                    options=('entropy_score',
                                            'least_confident_score',
                                            'least_margin_score'),
                                    format_func={'entropy_score': 'Entropy',
                                                'least_confident_score': 'Least Confidence',
                                                'least_margin_score': 'Least Margin'}.__getitem__,
                                    index=None,
                                    help = """
                **Entropy Score**: Entropy is the level of disorder or uncertainty in a
                given dataset or point, ranging from 0 to 1.

                **Least Confidence Score**: The confidence score represents the proability that
                the image was labeled correctly. Images with the lowest confidence scores
                will be displayed.

                **Least Margin Score**: The margin score quantifies the distance between 
                a single data point to the decision boundry. Images located close to
                the decision boundry will be diplayed.
                """)
        
    left_5, right_5 = st.columns(2)
    with left_5:
        lambda_vals = [round(x * 0.001, 3) for x in range(0, 1001)]
        session_lambda = st.select_slider(label="How often would you like to revalidate images?",
                                          options=lambda_vals,
                                          value=lambda_vals[69],
                                          help="""
                To ensure crowd surfed labels are accurate, labels should be validated
                more than once. This slider allows users to determine how quickly images
                are revalidated. 
                
                A larger value will lead to **less frequent** relabeling.

                A smaller value will lead to **more frequent** relabeling.""")

    with right_5:
        purpose = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        session_purpose = st.select_slider(label="For what purpose are the images being labeled?",
                                        options=purpose,
                                        format_func = {0.0: 'Retraining',
                                                        0.1: '0.1',
                                                        0.2: '0.2',
                                                        0.3: '0.3',
                                                        0.4: '0.4',
                                                        0.5: '0.5',
                                                        0.6: '0.6',
                                                        0.7: '0.7',
                                                        0.8: '0.8',
                                                        0.9: '0.9',
                                                        1.0: "Evaluation"}.__getitem__,
                                    help=""" 
                The purpose of this question is to determine the level of randomness.
                Moving the slider toward *Evaluation* will result in a more
                randomized selection of images while moving the slider toward 
                *Retraining* will ensure images with less certain labels are selected.""")

    left_6, right_6 = st.columns(2)
    with left_6:
        session_number = st.number_input(label='What is the prefered image batch size?',
                                min_value=0,
                                max_value=200,
                                value=100,
                                step=5)

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Image Validation</h4>""",
                unsafe_allow_html=True)
    
    # if user_number!=None and user_selec!=None and user_goal!=None:
    #     metrics_df = pd.read_parquet('data/inventory_df_with_scores.parquet.gzip')
    #     metrics_df['blob_name'] = metrics_df['image_path'].apply(lambda x: x.split('NAAMES/')[1])
        
    #     if user_goal == 'Model Retraining':
    #         filtered_df = metrics_df.sort_values(user_selec, ascending=True).head(user_number)
    #     elif user_goal == 'Performance Verification':
    #         filtered_df = metrics_df.sample(n=user_number)

    #     try:
    #         if 'image_iterator' not in st.session_state:
    #             st.session_state['image_iterator'] = data_utils.get_images(filtered_df, 1)
    #     except:
    #         st.error("Failed to instantiate class")

    # image_set = []
    # selection = {'entropy_score': 'Entropy',
    #              'least_confident_score': 'Least Confidence',
    #              'least_margin_score': 'Least Margin'}

    # if all((user_email!="", user_goal is not None, user_selec is not None, user_number is not None)):
    
    #     with st.form("plankton_form", clear_on_submit=True):   
    #         try:
    #             if 'image_iterator' in st.session_state:
    #                 image_set = next(st.session_state['image_iterator'])
    #         except:
    #             st.error("Failed to load next image set")

    #         for image in image_set: 

    #             st.image(image[0])
    #             ml_label = str(image[1][3])
            
    #             st.write("ML Generated Label: " + ml_label)
    #             st.write(selection[user_selec], " Score: ", image[1][user_selec])

    #             user_label = st.selectbox(
    #                 label = "Select the correct phytoplankton subcategory:",
    #                 options = ['Cilliate',
    #                             'Chloro',
    #                             'Crypto',
    #                             'Diatom',
    #                             'Dictyo',
    #                             'Dinoflagellate',
    #                             'Eugleno',
    #                             'Prymnesio',
    #                             'Other',
    #                             'Not phytoplanton'],
    #                 index=None
    #             )

    #         submitted = st.form_submit_button("Submit")
    #         if submitted:
    #             # Will this data be sent to a SQL Database?????????
    #             new_data = [image[1]['index'],
    #                     image[1]['image_path'],
    #                     image[1]['pred_label'],
    #                     image[1]['pred_class'],
    #                     image[1]['entropy_score'],
    #                     image[1]['least_confident_score'],
    #                     image[1]['least_margin_score'],
    #                     user_label,
    #                     user_name,
    #                     user_email,
    #                     user_lab,
    #                     user_date,
    #                     user_goal,
    #                     user_selec]
    #             st.write("The previous image was labeled: ", user_label)

