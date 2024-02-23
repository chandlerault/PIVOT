"""
Code that executes the contents of the Resources page
and is called by the main app.py script.

Functions:
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import streamlit as st

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Resources subpage. This
    page contains useful links and information for users labeling phytoplankton images.
    """

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Resources<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    # Display introduction to page
    st.markdown("""<h3 style='text-align: left; color: black;'>
                Phytoplankton Classification Resources</h4>""",
                unsafe_allow_html=True)

    st.write("""The following websites provide guidance with phytoplankton
             taxonomic identification that may be useful when labeling images
             using this tool.""")

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    # Display info from Kudela lab and center corresponding image
    st.markdown("""<li style='text-align:left; color:black; font-size:18px; font-weight:bold'>
                Phytoplankton Identification Guide by the Kudela Lab,
                University of California Santa Cruz:</li>""",
                unsafe_allow_html=True)

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    three_columns = st.columns([3,1,3])
    with three_columns[1]:
        st.image("images/tiny_drifters.jpeg")

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    st.write("""The Kudela lab has created a digital library that comprises images and
             distinctive characteristics of phytoplankton species discovered in the
             coastal waters of California, Oregon, and Washington. The online library
             can be found
             [here](http://oceandatacenter.ucsc.edu/PhytoGallery/phytolist.html).
             This digital library encompasses the information found in *Tiny Drifters*,
             a photo-based taxonomic guide to marine and freshwater phytoplankton in
             California.""")

    # Display info from Sosik lab and center corresponding image
    st.markdown("""<li style='text-align:left; color:black; font-size:18px; font-weight:bold'>
                Phytoplankton images labeled by the Sosik Lab at Woods Hole Oceanographic
                Institution:</li>""",
                unsafe_allow_html=True)

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    three_columns = st.columns(3)
    with three_columns[1]:
        st.image("images/WHOI_PrimaryLogo.png")

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    st.write("""A library of annotated plankton images for developing and evaluating
             classification methods was developed by the Sosik Lab at the Woods Hole
             Oceanographic Institution and provided as a website and downloadable data set.
             The website can be found
             [here](https://whoigit.github.io/whoi-plankton/index.html).
             The images found on this site were collected by an Imaging FlowCytobot, the
             same instrument used on the UTOPIA project.""")

    # Display info from University of Rhode Island and center corresponding image
    st.markdown("""<li style='text-align:left; color:black; font-size:18px; font-weight:bold'>
                Phytoplankton Wiki by researchers at the University of Rhode Island:</li>""",
                unsafe_allow_html=True)

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    three_columns = st.columns([3,1,3])
    with three_columns[1]:
        st.image("images/Rhode_Island.png")

    st.markdown("""<h3></h3>""", unsafe_allow_html=True)

    st.write("""An online gallery of labeled phytoplankton images was created by researchers at
             the University of Rhode Island. This gallery provides several example images
             for different phytoplankton categories, along with a description of their
             characteristics. The gallery can be found
             [here](https://virginie-sonnet.notion.site/Images-gallery-8898a8f7f1df439db5685c4641144a52).""")
