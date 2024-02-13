"""
This module manages the theme of the application, page navigation sidebar, subpages,
and provides a example phytoplankton images and references through a dropdown.

"""
import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from app_pages import image_validation
from app_pages import dashboard
from app_pages import resources
from app_pages import about

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
                 "Summary Metrics",
                 "Resources",
                 "About"],
        icons=['chevron-right', 'chevron-right', 'chevron-right', 'chevron-right'],
        default_index=0,
        styles={
        "container": {"padding": "0!important", "background-color": "#052e4a"},
        "icon": {"color": "white"},
        "nav-item": {"background-color": "#052e4a"},
        "nav-link": {"text-align": "left", "margin":"0px", "color": "white"},
        "nav-link-selected": {"background-color": "#052e4a"},
        }
    )

    # Add buttons to UTOPIA GitHub page
    st.markdown("""<h5 style='text-align: left; color: white;'>
                Links to relavent GitHub repositories:</h5>""",
                unsafe_allow_html=True)

    URL_STRING_GITHUB = "https://github.com/ifcb-utopia"
    st.markdown(
        # pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long
        f'<a href="{URL_STRING_GITHUB}" style="display: inline-block; width: 100%; padding-top: 5px; padding-bottom: 5px; background-color: #51bfff; font-weight: bold; color: black; text-align: center; border-radius: 4px;">ifcb-utopia \N{ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS}</a>',
        unsafe_allow_html=True)

    st.markdown("""<p style='text-align: left; color: white;'>
                </br>Select to view examples: </p>""",
                unsafe_allow_html=True)

    example_options = ["Chloro",
                       "Cilliate",
                       "Crypto",
                       "Diatom",
                       "Dictyo",
                       "Dinoflagellate",
                       "Eugleno",
                       "Unidentified",
                       "Prymnesio",
                       "Not Phytoplankton"]

    example_image = st.selectbox(options=example_options,
                                label_visibility ='collapsed',
                                label='examples',
                                index=None)
    if example_image == example_options[0]:
        st.write('Images HERE')
        st.write('Button to links HERE')

# Run specific app_page scrip when a page is selected from nav sidebar
if selected == "Image Validation":
    image_validation.main()
elif selected == "Summary Metrics":
    dashboard.main()
elif selected == "Resources":
    resources.main()
elif selected == "About":
    about.main()
