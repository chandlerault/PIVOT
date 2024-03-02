"""
This module manages the theme of the application, page navigation sidebar, subpages,
and provides a example phytoplankton images and references through a dropdown.

"""
import os
import glob
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from app_pages import image_validation
from app_pages import dashboard
from app_pages import resources
from app_pages import about
from app_pages import config_file

def read_example_images(file_path):
    for filename in glob.glob(file_path):
            image = Image.open(filename)
            st.image(image)

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
                 "About",
                 "Settings"],
        icons=['chevron-right', 'chevron-right', 'chevron-right', 'chevron-right', 'chevron-right'],
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

    URL_UTOPIA_GITHUB = "https://github.com/ifcb-utopia"
    st.markdown(
        # pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long
        f'<a href="{URL_UTOPIA_GITHUB}" style="display: inline-block; width: 100%; padding-top: 5px; padding-bottom: 5px; background-color: #51bfff; font-weight: bold; color: black; text-align: center; border-radius: 4px;">ifcb-utopia \N{ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS}</a>',
        unsafe_allow_html=True)
    
    URL_PIVOT_GITHUB = "https://github.com/chandlerault/PIVOT"
    st.markdown(
        # pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long
        f'<a href="{URL_PIVOT_GITHUB}" style="display: inline-block; width: 100%; padding-top: 5px; padding-bottom: 5px; background-color: #51bfff; font-weight: bold; color: black; text-align: center; border-radius: 4px;">PIVOT \N{ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS}</a>',
        unsafe_allow_html=True)

    st.markdown("""<p style='text-align: left; color: white;'>
                </br>Select to view examples: </p>""",
                unsafe_allow_html=True)

    # Set a list of phytoplankton to provide example images
    example_options = ["Chloro",
                       "Cilliate",
                       "Crypto",
                       "Diatom",
                       "Dictyo",
                       "Dinoflagellate",
                       "Eugleno",
                       "Unidentifiable",
                       "Prymnesio",
                       "Other"]

    # Display example images based on user selected phytoplankton
    example_image = st.selectbox(options=example_options,
                                label_visibility ='collapsed',
                                label='examples',
                                index=None)
    if example_image == example_options[0]:
        read_example_images('images/phytoplankton/chloro/*.png')
    elif example_image == example_options[1]:
        read_example_images('images/phytoplankton/ciliates/*.png')
    elif example_image == example_options[2]:
        read_example_images('images/phytoplankton/crypto/*.png')
    if example_image == example_options[3]:
        read_example_images('images/phytoplankton/diatoms/*.png')
    if example_image == example_options[4]:
        read_example_images('images/phytoplankton/dictyo/*.png')
    if example_image == example_options[5]:
        read_example_images('images/phytoplankton/dinoflagellates/*.png')
    if example_image == example_options[6]:
        read_example_images('images/phytoplankton/eugleno/*.png')
    if example_image == example_options[7]:
        st.write("""This category corresponds to images that contain phytoplankton
                    but the correct category cannot be identified due to blurred or
                    unclear image quality.""")
    if example_image == example_options[8]:
        read_example_images('images/phytoplankton/pyrmnesio/*.png')
    if example_image == example_options[9]:
        read_example_images('images/phytoplankton/other/*.png')

# Run specific app_page scrip when a page is selected from nav sidebar
if selected == "Image Validation":
    if os.stat("config/config.yaml").st_size == 0:
        image_validation.header()
        st.error("""No database configuration found.
                 Please enter the database configuration information
                 below.""")
        config_file.main()
    else:
        image_validation.header()
        image_validation.main()
elif selected == "Summary Metrics":
    dashboard.main()
elif selected == "Resources":
    resources.main()
elif selected == "About":
    about.main()
elif selected == "Settings":
    config_file.header()
    config_file.main()
