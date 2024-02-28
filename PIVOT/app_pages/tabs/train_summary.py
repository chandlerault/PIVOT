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
    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    if model_pred.empty:
        st.error("Please select at LEAST one phytoplankton to view.")
    else:
        three_columns = st.columns([5,.2,5])
        with three_columns[0]:
            st.plotly_chart(ds.plot_confusion_matrix(model_pred,
                                            ['true_label', 'pred_label'],
                                            classes=class_labels,
                                            normalize=True), use_container_width=True)
            c_report = ds.get_classification_report(model_pred,
                                                    ['true_label','pred_label'],
                                                    class_names = class_labels)

        with three_columns[2]:
            c_report = c_report.sort_values(by=['f1-score'], ascending=False)
            st.plotly_chart(ds.plot_roc_curve(model_pred['true_label'],
                                              model_pred.iloc[:, 5:15], 
                                              class_labels), use_container_width=True)
        st.plotly_chart(ds.plot_precision_recall_f1(c_report),
                        use_container_width=True)
