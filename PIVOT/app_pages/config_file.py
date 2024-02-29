"""
Code that executes the contents of the Settings page
and is called by the main app.py script and the
Image Validation page.

Functions:
    - header: Executes the Streamlit formatted HTML header/banner for the page.
    - main: Executes the Streamlit formatted HTML when called by app.py,
            not including the header.
"""
import os
import streamlit as st

from utils import load_config

def header():
    """
    Executes the Streamlit formatted HTML banner for the page.
    """
    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Settings<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

def main():
    """
    The user inputs here overwrite the values saved in the config file found
    in config/config.yaml.
    """
    # Create form for configuration input variables
    with st.form('config_from', clear_on_submit=True, border=False):

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    Blob Storage</h3>""",
                    unsafe_allow_html=True)

        # Prompt user for Blog Storage info
        connection_string = st.text_input(label="Connection String:", key='connection_string')
        image_container = st.text_input("Image Container:", key='image_container')

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    SQL Database</h3>""",
                    unsafe_allow_html=True)

        # Prompt user for SQL Database info
        left_1, right_1 = st.columns(2)
        with left_1:
            server = st.text_input("Server Name:", key='server')
            db_user = st.text_input("Admin Username:", key='db_user')
        with right_1:
            database = st.text_input("Database Name:", key='database')
            db_password = st.text_input("Password:", key='db_password')

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    Azure ML Model</h3>""",
                    unsafe_allow_html=True)

        left_2, right_2 = st.columns(2)
        with left_2:
            subscription_id = st.text_input(label="Subscription ID:", key="subscription_id")
            resource_group = st.text_input("Resource Group:", key='resource_group')
            workspace_name = st.text_input("Workspace Name:", key='workspace_name')
            experiment_name = st.text_input('Experiment Name:', key='experiment_name')
        with right_2:
            api_key = st.text_input("Model API Key:", key='api_key')
            model_name = st.text_input("Model Name:", key='model_name')
            endpoint_name = st.text_input("Model Endpoint Name:", key='endpoint_name')
            deployment_name = st.text_input("Model Deployment Name:", key='deployment_name')

        st.markdown("""<h1></h1>""", unsafe_allow_html=True)

        # Display current database configuration variables if they exist
        if os.stat("config/config.yaml").st_size != 0:
            with st.expander("Current Database Configuration:"):
                config_dict = load_config()
                st.write("*Connection String:*", config_dict['connection_string'])
                st.write("*Image Container:*", config_dict['image_container'])
                st.write("*Server Name:*", config_dict['server'])
                st.write("*Database Name:*", config_dict['database'])
                st.write("*Admin Username:*", config_dict['db_user'])
                st.write("*Password:*", config_dict['db_password'])
                st.write("*Subscription ID:*", config_dict['subscription_id'])
                st.write("*Resource Group:*", config_dict['workspace_name'])
                st.write("*Workspace Name:*", config_dict['db_password'])
                st.write("*Experiment Name:*", config_dict['experiment_name'])
                st.write("*Model API Key:*", config_dict['api_key'])
                st.write("*Model Name:*", config_dict['model_name'])
                st.write("*Model Endpoint Name:*", config_dict['endpoint_name'])
                st.write("*Model Deployment Name:*", config_dict['deployment_name'])
            st.warning("""Warning! By submitting, you will overwrite your
                       database configurations.""")

        # Open the config.yaml file and write user inputted variables
        if st.form_submit_button("Submit"):
            with open("config/config.yaml", "w", encoding="utf-8") as file:
                file.write("connection_string: " + connection_string + "\n")
                file.write("image_container: " + image_container + "\n")
                file.write("server: " + server + "\n")
                file.write("database: " + database + "\n")
                file.write("db_user: " + db_user + "\n")
                file.write("db_password: " + db_password + "\n")
                file.write("subscription_id: " + subscription_id + "\n")
                file.write("resource_group: " + resource_group + "\n")
                file.write("workspace_name: " + workspace_name + "\n")
                file.write("experiment_name: " + experiment_name + "\n")
                file.write("api_key: " + api_key + "\n")
                file.write("model_name: " + model_name + "\n")
                file.write("endpoint_name: " + endpoint_name + "\n")
                file.write("deployment_name: " + deployment_name + "\n")
            st.rerun()
