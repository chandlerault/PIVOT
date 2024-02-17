"""
Code that executes the contents of the Image Validation page
and is called by the main app.py script.

Functions:
    - get_user_experience: Calculates the experience of a user based on the number
                           number of images they have labeled and their domain experience.
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import os
import streamlit as st
import pandas as pd

from utils import app_utils
from utils import sql_utils

def get_user_experience(num_labels, domain):
    """
    This function calculates the experience of a user from a range of 1 to 5, where
    1 indicates no experience and 5 indicates an expert.

    Args:
        - num_labels (str): User prompted range indicating the number of images labeled.
        - domain (str): User prompted response determine their domain.

    Returns:
        - exp_level (int): The level of experience ranging from 1 to 5.
    """
    exp_level = 1

    if domain == "No" and num_labels == "None":
        exp_level = 1
    elif domain == "No" and num_labels == "25 to 100":
        exp_level = 2
    elif domain == "No" and num_labels == "100 to 500":
        exp_level = 3
    elif domain == "No" and num_labels == "500 to 1000":
        exp_level = 4
    elif domain == "No" and num_labels == "1000+":
        exp_level = 5
    elif domain == "Yes" and num_labels == "None":
        exp_level = 2
    elif domain == "Yes" and num_labels == "25 to 100":
        exp_level = 3
    elif domain == "Yes" and num_labels == "100 to 500":
        exp_level = 4
    elif domain == "Yes" and num_labels == "500 to 1000":
        exp_level = 5
    elif domain == "Yes" and num_labels == "1000+":
        exp_level = 5

    return exp_level

def header():
    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Image Validation subpage. Users
    are prompted to fill out several forms used to determine the images they will label. All
    forms link to the SQL Database using functions from both sql_utils and app_utils from
    the utils folder.
    """    
    st.markdown("""<h3 style='text-align: left; color: black;'>
            User Information</h3>""",
            unsafe_allow_html=True)
    
    user_account = []

    left_1, right_1 = st.columns(2)
    with left_1:
        user_email = st.text_input(label = "Please enter your email:")
    with right_1:
        pass

    if user_email != '':
        user_account = app_utils.get_user(user_email)

        if user_account is None:
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
                user_num_labels = st.radio(
                    label="Approximately how many images have you labeled?",
                    options=['None', '25 to 100',
                            '100 to 500', '500 to 1000',
                            '1000+'],
                            index=None)
                user_experience = get_user_experience(user_num_labels, user_domain)

            new_user = {
                'email': user_email,
                'name': user_name,
                'experience': user_experience,
                'lab': user_lab
            }

            user_confirm = st.button(label="Submit", key="user_button")

            if user_confirm:
                app_utils.create_user(new_user)
                st.write("User Added!")
                user_account = app_utils.get_user(user_email)
        elif user_account is not None:
            st.markdown("""<h5 style='text-align: left; color: black;'>
                User Found</h5>""",
                unsafe_allow_html=True)
            st.write('Name: ' + str(user_account['name']))
            st.write('Experience: ' + str(user_account['experience']))
            st.write('Lab: ' + str(user_account['lab']))
            st.write('Email: ' + str(user_account['email']))

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Session Specifications</h4>""",
                unsafe_allow_html=True)

    label_df = pd.DataFrame()

    models = app_utils.get_models()

    if not models:
        st.error("""Please ensure database configuration information is correct. If so,
                 restart app to ensure database configurations have been
                 saved with the following command.""")
        st.code("streamlit run app.py", language=None)
    else:
        left_4, right_4 = st.columns(2)
        with left_4:
            model_dic = {}
            model_names = []

            for i in range(1,len(models)):
                model_names.append(models[i]['m_id'])
                model_dic[models[i]['m_id']] = models[i]['model_name']

            session_model = st.selectbox(label='Select the model you wish to validate:',
                                        options=tuple(model_names),
                                        format_func=model_dic.__getitem__,
                                        index=None)

        with right_4:
            dissimilarities = app_utils.get_dissimilarities()
            diss_dic = {}
            diss_names = []

            for j in range(1, len(dissimilarities)):
                diss_names.append(dissimilarities[j]['d_id'])
                diss_dic[dissimilarities[j]['d_id']] = dissimilarities[j]['name']

            session_dissim = st.selectbox(label='What selection method would you like to use?',
                                        options=tuple(diss_names),
                                        format_func=diss_dic.__getitem__,
                                        index=None,
                                        help = """
                    **Entropy Score**: Entropy is the level of disorder or uncertainty in a
                    given dataset or point, ranging from 0 to 1.

                    **Least Confident Score**: The confident score represents the proability that
                    the image was labeled correctly. Images with the lowest confidence scores
                    will be displayed.

                    **Least Margin Score**: The margin score quantifies the distance between
                    a single data point to the decision boundry. Images located close to
                    the decision boundry will be diplayed.
                    """)

        left_5, right_5 = st.columns(2)
        with left_5:
            session_number = st.number_input(label='What is the prefered image batch size?',
                                    min_value=0,
                                    max_value=200,
                                    value=None,
                                    step=5)
        with right_5:
            pass

        with st.expander("Advanced Specifications"):
            left_6, right_6 = st.columns(2)
            with left_6:
                lambda_vals = [round(x * 0.001, 3) for x in range(0, 1001)]
                session_lambda = st.select_slider(
                                label="How often would you like to revalidate images?",
                                options=lambda_vals,
                                value=lambda_vals[69],
                                help="""
                        To ensure crowd surfed labels are accurate, labels should be validated
                        more than once. This slider allows users to determine how quickly images
                        are revalidated. 
                
                        A larger value will lead to **less frequent** relabeling.

                        A smaller value will lead to **more frequent** relabeling.""")

            with right_6:
                purpose = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                session_purpose = st.select_slider(
                    label="For what purpose are the images being labeled?",
                    options=purpose,
                    value=purpose[10],
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

        if session_model is not None and session_dissim is not None and session_number is not None:
            label_df = sql_utils.get_label_rank_df(model_id=session_model,
                                                dissimilarity_id=session_dissim,
                                                batch_size=session_number,
                                                relabel_lambda=session_lambda,
                                                random_ratio=session_purpose)
            st.write("Retreived Images!")

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    Image Validation</h3>""",
                    unsafe_allow_html=True)

        new_df = pd.DataFrame(columns=['i_id', 'u_id', 'weight', 'label'])

        if label_df is not None:
            if not label_df.empty:
                with st.form('image_validation_form', clear_on_submit=True):
                    for count in range(0, len(label_df)):
                        widget_selectbox = 'plankton_select_' + str(count)
                        widget_checkbox = 'plankton_check_' + str(count)

                        label_image = app_utils.get_image(label_df.iloc[count]['BLOB_FILEPATH'])
                        st.image(label_image)

                        label_pred = label_df.iloc[count]['PRED_LABEL']
                        st.write('ML Generated Label: ', label_pred)

                        label_id = label_df.iloc[count]['IMAGE_ID']
                        st.write('Image ID: ', label_id)

                        label_probs = label_df.iloc[count]['PROBS']
                        label_probs = pd.DataFrame.from_dict(label_probs, orient='index')
                        column_name = label_probs.columns[0]
                        label_probs = label_probs.rename(columns={column_name: "PROBS"})
                        label_probs = label_probs.sort_values(by='PROBS', ascending=False)
                        label_probs_options = label_probs.index.values.tolist()

                        user_label = st.selectbox(
                            label="Select the correct phytoplankton subcategory:",
                            key=widget_selectbox,
                            options = label_probs_options)

                        user_add = st.checkbox(label='Confirm label',
                                            key=widget_checkbox,
                                            value=False)
                        if user_add and not user_account:
                            st.error("Please submit your user information!")
                        elif user_add and user_account:
                            new_df.loc[count] = [label_df.iloc[count]['IMAGE_ID'],
                                                user_account['u_id'],
                                                user_account['experience'],
                                                user_label]

                        st.divider()
                    submitted = st.form_submit_button("Submit")
                    if submitted and user_account:
                        st.markdown("""<h5 style='text-align: left; color: black;'>
                            Your responses have been recorded!</h5>""",
                            unsafe_allow_html=True)
                        app_utils.insert_label(new_df)
                        st.rerun()
                    elif submitted and not user_account:
                        st.markdown("""<h5 style='text-align: left; color: black;'>
                        Please resubmit once your user information has been recorded.</h5>""",
                        unsafe_allow_html=True)
        else:
            st.error("""No images match the specification.""")
