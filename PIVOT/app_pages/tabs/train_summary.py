"""
Code that executes the contents of the Summary Metrics: Train Summary tab
and is called by the main dashboard.py script. This file formats data with the
aid of the dashboard_utils.py file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    
    with st.container(border=True):

        st.markdown("""<h3 style='text-align: left; color: black;'>
                Train Summary Dashboard</h3>""",
                unsafe_allow_html=True)
        
        st.write("""This interactive dashboard shows the summary performance metrics
                 of the CNN model trained on almost 400,000 phytoplankton images. Filter
                 the different graphs by selecting items on the legends, click and drag
                 over graphs to zoom, and download graphs as PNGs by hovering over graphs
                 and selecting the camera icon.""")
        
        model_pred['high_group'] = model_pred['high_group'].fillna('Unidentifiable')
        group_count = model_pred.groupby('high_group')['true_label'] \
                                .count().sort_values(ascending=True)
        
        st.markdown("""<h1></h1>""", unsafe_allow_html=True)

        two_columns = st.columns([.75,5])
        with two_columns[0]:
            st.metric("Accuracy:", f"{model_stats[0]*100:.2f} %")
            st.metric("Precision:", f"{model_stats[1]*100:.2f} %")
            st.metric("Recall:", f"{model_stats[2]*100:.2f} %")
            st.metric("Images Validated:", f"{len(model_pred)}")

        with two_columns[1]:
            if model_pred.empty:
                st.error("Please select at LEAST one phytoplankton to view.")
            else:
                nested_two_columns = st.columns(2)
                with nested_two_columns[0]:
                    st.plotly_chart(ds.plot_confusion_matrix(model_pred,
                                                    ['true_label', 'pred_label'],
                                                    classes=class_labels,
                                                    normalize=True), use_container_width=True)
                    c_report = ds.get_classification_report(model_pred,
                                                            ['true_label','pred_label'],
                                                            class_names = class_labels)

                with nested_two_columns[1]:
                    st.plotly_chart(ds.plot_roc_curve(model_pred['true_label'],
                                                    model_pred.iloc[:, 5:15], 
                                                    class_labels), use_container_width=True)
                    
        two_columns = st.columns([4,3])
        with two_columns[0]:
            c_report = c_report.sort_values(by=['f1-score'], ascending=False)
            st.plotly_chart(ds.plot_precision_recall_f1(c_report),
                            use_container_width=True)
        with two_columns[1]:
            fig = px.bar(group_count,
                     x='true_label',
                     y=group_count.index,
                     orientation='h',
                     labels={
                         "true_label": "",
                         "high_group": ""})
            fig.update_layout(title_text='<i><b>Class Distribution</b></i>')
            fig.update_traces(marker_color='#094789')
            fig.update_xaxes(showgrid=True)
            fig.update_traces(hovertemplate=None)
            fig.update_layout(hovermode="x")
            st.plotly_chart(fig, use_container_width=True)
