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
        connection_string = st.text_input(label="Connection String:")
        image_container = st.text_input("Image Container")

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    SQL Database</h3>""",
                    unsafe_allow_html=True)

        # Prompt user for SQL Database info
        left_1, right_1 = st.columns(2)
        with left_1:
            server = st.text_input("Server Name:")
            db_user = st.text_input("Admin Username:")
        with right_1:
            database = st.text_input("Database Name:")
            db_password = st.text_input("Password:")

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
            st.rerun()
