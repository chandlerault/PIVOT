"""
Code that executes the contents of the Summary Metrics: Test Summary tab
and is called by the main dashboard.py script. This file formats data with the
aid of the dashboard_utils.py file.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

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

            validated_df = sql_utils.get_test_set_df(selected_model_sum)
            if validated_df is not None:
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

                three_columns= st.columns(3)
                with three_columns[0]:
                    st.metric("Accuracy:",
                                f"{val_stats[0]*100:.2f} %",
                                delta=f"{(val_stats[0] - model_stats[0])*100:.2f} %")
                with three_columns[1]:
                    st.metric("Precision:",
                                f"{val_stats[1]*100:.2f} %",
                                delta=f"{(val_stats[1] - model_stats[1])*100:.2f} %")
                with three_columns[2]:
                    st.metric("Recall:",
                                f"{val_stats[2]*100:.2f} %",
                                delta=f"{(val_stats[2] - model_stats[2])*100:.2f} %")

                st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                five_columns = st.columns([.5,2,.5,2,.5])
                with five_columns[1]:
                    c_report_test = ds.get_classification_report(
                                    validated_df,
                                    ['CONSENSUS', 'PRED_LABEL', None])
                    st.plotly_chart(ds.plot_precision_recall(c_report_test),
                              use_container_width=True)

                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    st.pyplot(ds.plot_confusion_matrix(validated_df,
                                                        ['CONSENSUS', 'PRED_LABEL'],
                                                        classes=c_report_test.index,
                                                        normalize=True))
                with five_columns[3]:
                    c_report_test = c_report_test.sort_values(by=['f1-score'],
                                                              ascending=False)
                    st.plotly_chart(ds.plot_f1_score(c_report_test),
                                               use_container_width=True)
