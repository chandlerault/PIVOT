import os
import streamlit as st

def header():

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Settings<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

def main():

    with st.form('config_from', clear_on_submit=False, border=False):

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    Blob Storage</h3>""",
                    unsafe_allow_html=True)

        connection_string = st.text_input(label="Connection String:")
        image_container = st.text_input("Image Container")

        st.divider()

        st.markdown("""<h3 style='text-align: left; color: black;'>
                    SQL Database</h3>""",
                    unsafe_allow_html=True)
        
        left_2, right_2 = st.columns(2)
        with left_2:
            server = st.text_input("Server Name:")
            db_user = st.text_input("Admin Username:")
        with right_2:
            database = st.text_input("Database Name:")
            db_password = st.text_input("Password:")

        if os.stat("config/config.yaml").st_size != 0:
            st.warning("Warning! By submitting, you will overwrite your database configurations.")

        if st.form_submit_button("Submit"):
            with open("config/config.yaml", "w") as f:
                    f.write("connection_string: " + connection_string + "\n")
                    f.write("image_container: " + image_container + "\n")
                    f.write("server: " + server + "\n")
                    f.write("database: " + database + "\n")
                    f.write("db_user: " + db_user + "\n")
                    f.write("db_password: " + db_password + "\n")
            st.markdown("""<h6 style='text-align: left; color: black;'>
                        Reload Streamlit by running the following command to 
                        complete the configuration update.</h6>""",
                        unsafe_allow_html=True)
            st.code("streamlit run app.py", language=None)