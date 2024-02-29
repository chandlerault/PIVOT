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
                st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                three_columns = st.columns([5,.2,5])
                with three_columns[0]:
                    c_report_test = ds.get_classification_report(
                                    validated_df,
                                    ['CONSENSUS', 'PRED_LABEL', None])

                    st.plotly_chart(ds.plot_confusion_matrix(validated_df,
                                                ['CONSENSUS', 'PRED_LABEL'],
                                                classes=c_report_test.index,
                                                normalize=True), use_container_width=True)

                st.plotly_chart(ds.plot_precision_recall_f1(c_report_test),
                                use_container_width=True)

                with three_columns[2]:
                    c_report_test = c_report_test.sort_values(by=['f1-score'],
                                                              ascending=False)
                    
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
                    
### Aditi
                class_labels = ["Chloro",
                "Ciliate",
                "Crypto",
                "Diatom",
                "Dictyo",
                "Dinoflagellate",
                "Eugleno",
                "Other",
                "Prymnesio",
                "Unidentifiable"]

                model_preds = pd.read_csv('data/model-summary-cnn-v1-b3.csv')
                pred_label_counts = model_preds.groupby('pred_label').size().reset_index(name='count')['count'].values

                val_label_counts = validated_df.groupby('PRED_LABEL').size().reset_index(name='count')
                val_label_counts = [val_label_counts[val_label_counts.PRED_LABEL == label]['count'].values[0] if len(val_label_counts[val_label_counts.PRED_LABEL == label]) > 0 else 0 for label in class_labels]

                stacked_df = pd.DataFrame({'class': class_labels,
                                           'total': [100 for i in range(10)],
                                           'val': (val_label_counts/pred_label_counts)*100})
                
                custom_colors = ['#0B5699', '#EDF8E6']
                fig = px.bar(stacked_df,
                             x='class',
                             y=['val', 'total'],
                             color_discrete_sequence=custom_colors)
                fig.update_layout(title_text='<i><b>Title of Stacked Bar Chart</b></i>')

                st.plotly_chart(fig)
                
                custom_colors = ['#0B5699']
                fig = px.bar(stacked_df,
                             x='class',
                             y='val',
                             color_discrete_sequence=custom_colors)
                fig.update_layout(title_text='<i><b>Title of Normal Bar Chart</b></i>')

                st.plotly_chart(fig)
