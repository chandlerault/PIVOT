"""
Code that executes the contents of the Summary Metrics: Test Summary tab
and is called by the main dashboard.py script. This file formats data with the
aid of the dashboard_utils.py file.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import sql_utils
from utils import app_utils

from utils import dashboard_utils as ds

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Test Summary tab and
    displays summary statistics and graphs for the test model. Users can select from
    the different test models stored on the SQL Database.
    """
    if os.stat("config/config.yaml").st_size == 0:
        st.error("""No database configuration found. Please update the database
                 configuration in the Settings page.""")
    else:
        model_list = app_utils.get_models()
        if not model_list:
            st.error("""Please ensure database configuration information is correct
                        and update on the Settings page.""")
        else:          
            two_columns = st.columns(2)
            with two_columns[0]:
                model_list = app_utils.get_models()
                model_dictionary = {}
                model_name = []

                for i in range(1,len(model_list)):
                    model_name.append(model_list[i]['m_id'])
                    model_dictionary[model_list[i]['m_id']] = model_list[i]['model_name']

                selected_model_sum = st.selectbox(
                    label='Select the model you wish to evaluate:',
                    options=tuple(model_name),
                    format_func=model_dictionary.__getitem__,
                    index=None)

            st.markdown("""<h1></h1>""", unsafe_allow_html=True)

            if selected_model_sum:
                validated_df = sql_utils.get_test_set_df(selected_model_sum)
            else:
                validated_df = None

            if validated_df is not None:

                with st.container(border=True):

                    st.markdown("""<h3 style='text-align: left; color: black;'>
                            Test Summary Dashboard</h3>""",
                            unsafe_allow_html=True)
                    
                    st.write("""This interactive dashboard shows the summary performance
                            metrics of the CNN on an unseen dataset. Filter the different
                            graphs by selecting items on the legends, click and drag over
                            graphs to zoom, and download graphs as PNGs by hovering over
                            graphs and selecting the camera icon.""")
                    
                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    with st.expander("View Labeling Progress:"):
                        st.write('Aditis graphs here')

                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    validated_df['IS_CORRECT'] = (validated_df['PRED_LABEL'] == \
                                                validated_df['CONSENSUS']).astype(int)
                    val_stats = ds.get_acc_prec_recall(validated_df,
                                                    ['IS_CORRECT',
                                                        'CONSENSUS',
                                                        'PRED_LABEL'])

                    model_stats = ds.get_acc_prec_recall(
                        pd.read_csv('data/model-summary-cnn-v1-b3.csv'),
                        ['is_correct',
                        'true_label',
                        'pred_label'])

                    two_columns = st.columns([.75,5])
                    with two_columns[0]:
                        st.metric("Accuracy:",
                                    f"{val_stats[0]*100:.2f} %",
                                    delta=f"{(val_stats[0] - model_stats[0])*100:.2f} %")
                        st.metric("Precision:",
                                    f"{val_stats[1]*100:.2f} %",
                                    delta=f"{(val_stats[1] - model_stats[1])*100:.2f} %")
                        st.metric("Recall:",
                                    f"{val_stats[2]*100:.2f} %",
                                    delta=f"{(val_stats[2] - model_stats[2])*100:.2f} %")

                    with two_columns[1]:
                        nested_two_columns = st.columns(2)
                        with nested_two_columns[0]:
                            c_report_test = ds.get_classification_report(
                                            validated_df,
                                            ['CONSENSUS', 'PRED_LABEL', None])

                            st.plotly_chart(ds.plot_confusion_matrix(validated_df,
                                                ['CONSENSUS', 'PRED_LABEL'],
                                                classes=c_report_test.index,
                                                normalize=True), use_container_width=True)
                        with nested_two_columns[1]:
                            st.plotly_chart(ds.plot_roc_curve(validated_df['CONSENSUS'],
                                            pd.DataFrame(validated_df['PROBS'].tolist()), 
                                            c_report_test.index.sort_values(ascending=True)),
                                            use_container_width=True)
                    two_columns = st.columns([4,3])
                    with two_columns[0]:
                        c_report_test = c_report_test.sort_values(by=['f1-score'],
                                                                        ascending=False)
                        st.plotly_chart(ds.plot_precision_recall_f1(c_report_test),
                                    use_container_width=True)
                    with two_columns[1]:                    
                        agg_df = validated_df.groupby(['PRED_LABEL', 'CONSENSUS']).size().reset_index(name='count')

                        fig = px.sunburst(agg_df, path=['PRED_LABEL', 'CONSENSUS'], values='count')
                        fig.update_traces(
                                            marker_colors=[
                                                px.colors.qualitative.Prism[c] for c in pd.factorize(fig.data[0].labels)[0]
                                            ],
                                            leaf_opacity=.8,
                                        )
                        fig.update_layout(title_text='<i><b>Sunburst Plot</b></i>')
                        st.plotly_chart(fig, use_container_width=True)
