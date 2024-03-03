"""
Code that executes the contents of the Summary Metrics page
and is called by the main app.py script. The content for each tab can be found
in the tabs/ folder.
"""
import streamlit as st

from app_pages.tabs import train_summary
from app_pages.tabs import test_summary

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Dashboard subpage and
    displays summary statistics and graphs for the trained model and test model.
    Users can select from the different test models stored on the SQL Database.
    """
    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Summary Metrics<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.empty()

    tab_1, tab_2 = st.tabs(['Train Summary', 'Test Summary'])
    with tab_1:
        train_summary.main()
    with tab_2:
        test_summary.main()
