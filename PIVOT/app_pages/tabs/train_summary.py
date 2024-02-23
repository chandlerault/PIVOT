"""
Code that executes the contents of the Summary Metrics: Train Summary tab
and is called by the main dashboard.py script. This file formats data with the
aid of the dashboard_utils.py file.
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils import dashboard_utils as ds

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Train Summary tab and
    displays summary statistics and graphs for the trained model. Data is read in as
    a CSV from the data/ folder.
    """
    class_labels = ["Chloro",
                    "Ciliate",
                    "Crypto",
                    "Diatom",
                    "Dictyo",
                    "Dinoflagellate",
                    "Eugleno",
                    "Unidentifiable",
                    "Prymnesio",
                    "Other"]

    model_pred = pd.read_csv('data/model-summary-cnn-v1-b3.csv')
    model_stats = ds.get_acc_prec_recall(model_pred, ['is_correct',
                                                    'true_label',
                                                    'pred_label'])

    three_columns = st.columns(3)
    with three_columns[0]:
        st.metric("Accuracy:", f"{model_stats[0]*100:.2f} %")
    with three_columns[1]:
        st.metric("Precision:", f"{model_stats[1]*100:.2f} %")
    with three_columns[2]:
        st.metric("Recall:", f"{model_stats[2]*100:.2f} %")

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.markdown("""<h6 style='text-align: left; color: black;'>
                Filter Phytoplankton Subcategories</h6>""",
                unsafe_allow_html=True)

    filtered_phyto = list(range(0,10,1))

    check_box_columns = st.columns(5)
    with check_box_columns[0]:
        filtered_phyto[0] = st.checkbox(class_labels[0], value=1)
        filtered_phyto[5] = st.checkbox(class_labels[5], value=1)
    with check_box_columns[1]:
        filtered_phyto[1] = st.checkbox(class_labels[1], value=1)
        filtered_phyto[6] = st.checkbox(class_labels[6], value=1)
    with check_box_columns[2]:
        filtered_phyto[2] = st.checkbox(class_labels[2], value=1)
        filtered_phyto[7] = st.checkbox(class_labels[7], value=1)
    with check_box_columns[3]:
        filtered_phyto[3] = st.checkbox(class_labels[3], value=1)
        filtered_phyto[8] = st.checkbox(class_labels[8], value=1)
    with check_box_columns[4]:
        filtered_phyto[4] = st.checkbox(class_labels[4], value=1)
        filtered_phyto[9] = st.checkbox(class_labels[9], value=1)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    count = 0
    for idx, item in enumerate(filtered_phyto):
        if item is False:
            model_pred = model_pred[model_pred['true_label'] != idx]
            model_pred = model_pred[model_pred['pred_label'] != idx]
            del class_labels[idx-count]
            count += 1

    if model_pred.empty:
        st.error("Please select at LEAST one phytoplankton to view.")
    else:
        five_columns = st.columns([.5,2,.5,2,.5])
        with five_columns[1]:
            c_report = ds.get_classification_report(model_pred,
                                                    ['true_label','pred_label'],
                                                    class_names = class_labels)
            st.plotly_chart(ds.plot_precision_recall(c_report),
                            use_container_width=True)

            st.markdown("""<h1></h1>""", unsafe_allow_html=True)

            st.pyplot(ds.plot_confusion_matrix(model_pred,
                                            ['true_label', 'pred_label'],
                                            classes=class_labels,
                                            normalize=True))
        with five_columns[3]:
            c_report = c_report.sort_values(by=['f1-score'], ascending=False)
            st.plotly_chart(ds.plot_f1_score(c_report),
                            use_container_width=True)
