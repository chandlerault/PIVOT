import os

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from app_pages import image_validation
from app_pages import dashboard

# Set current working directory to .
cwd = os.getcwd()
base_path = cwd.split("PIVOT", maxsplit=1)[0]
curr_path = base_path + \
    'PIVOT/PIVOT/'
os.chdir(curr_path)

st.set_page_config(layout="wide")

# Set color and style of website
st.markdown("""<style>
            [data-testid="stHeader"] {
                background-color: #052e4a;
                color: white;
            }
            [data-testid="collapsedControl"] {
                color: white;
            }
            [data-testid=stSidebar] {
                background-color: #052e4a !important;
                color: white
            }
            div.stButton > button:first-child {
                background-color: #51bfff;
                color: black;
            }
            div.stDownloadButton > button:first-child {
                background-color: #51bfff;
                color: #ffffff;
            }
            </style>""", unsafe_allow_html=True)

# Configure navigation sidebar
with st.sidebar:

    logo = Image.open('images/apl-uw_logo-over.png')
    left, right = st.columns(2)
    with left:
        st.image(logo)

    # Set different sub page options
    selected = option_menu(
        menu_title="",
        options=["Image Validation",
                 "Summary Metrics"],
        icons=['chevron-right', 'chevron-right'],
        default_index=0,
        styles={
        "container": {"padding": "0!important", "background-color": "#052e4a"},
        "icon": {"color": "white"},
        "nav-item": {"background-color": "#052e4a"},
        "nav-link": {"text-align": "left", "margin":"0px", "color": "white"},
        "nav-link-selected": {"background-color": "#052e4a"},
        }
    )

    # Add buttons to Safety Data Sheet and Hazard Assesmment external links
    st.markdown("""<h5 style='text-align: left; color: white;'>
                This is a space where we can link important information.</h5>""",
                unsafe_allow_html=True)
    URL_STRING_SDS = "https://google.com"
    st.markdown(
        f'<a href="{URL_STRING_SDS}" style="display: inline-block; width: 100%; padding-top: 5px; padding-bottom: 5px; background-color: #51bfff; font-weight: bold; color: black; text-align: center; border-radius: 4px;">Some Link! \N{ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS}</a>',
        unsafe_allow_html=True)

# Run specific app_page scrip when a page is selected from nav sidebar
if selected == "Image Validation":
    image_validation.main()

if selected == "Summary Metrics":

    dashboard.main()
