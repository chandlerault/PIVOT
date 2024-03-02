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
import plotly.graph_objects as go

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

                    st.markdown("""<h2 style='text-align: left; color: black;'>
                            Test Summary Dashboard</h2>""",
                            unsafe_allow_html=True)
                    
                    st.write("""This interactive dashboard shows the summary performance
                            metrics of the CNN on an unseen dataset. Filter the different
                            graphs by selecting items on the legends, click and drag over
                            graphs to zoom, and download graphs as PNGs by hovering over
                            graphs and selecting the camera icon.""")
                    
                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    with st.expander("View Labeling Progress:"):
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

                        percent_df = pd.DataFrame({'class': class_labels,
                                               'total': [100 for i in range(10)],
                                               '% Images Labeled': (val_label_counts/pred_label_counts)*100})

                        count_df = pd.DataFrame({'class': class_labels,
                                                 '# Images Labeled': val_label_counts})

                        def update_plot(count_df, target):
                            count_df['remaining'] = target - count_df['# Images Labeled']
                            count_df['color'] = ['red' if i > 0 else 'green' for i in count_df['remaining']]
                            color_seq = {'red': 'red', 'green':'green'}
                            count_df.loc[count_df['remaining'] < 0, 'remaining'] = 0
                            fig = px.bar(count_df,
                                         x='class',
                                         y='# Images Labeled',
                                         labels={'remaining': 'Remaining Images'},
                                         color='color',
                                         color_discrete_map=color_seq)

                            hover_template = ("<b>%{x}</b><br>"
                                              "Labeled: %{y}<br>"
                                              "Remaining: %{customdata}<br>"
                                              "<extra></extra>")

                            fig.update_traces(hovertemplate=hover_template,
                                              customdata=count_df['remaining'],
                                              showlegend=False)

                            fig.add_hline(y=target,
                                          line_dash="dash",
                                          annotation_text=f'Target: {target}',
                                          annotation_position="top left")

                            fig.add_trace(go.Bar(x=count_df['class'],
                                                 y=target - count_df['# Images Labeled'],
                                                 base=count_df['# Images Labeled'],
                                                 customdata=count_df['remaining'],
                                                 marker=dict(color='rgba(255, 0, 0, 0.1)'),
                                                 showlegend=False, 
                                                 hovertemplate= ("<b>%{x}</b><br>"
                                                                  "Remaining: %{customdata}<br>"
                                                                  "<extra></extra>")))

                            fig.update_layout(title_text=f'<i><b>Number of Images Labeled per Class (Target: {target} images)',
                                              xaxis_title="Class",
                                              yaxis_title="# Images Labeled",
                                              hovermode='closest')

                            return fig
                        three_columns = st.columns([5, .2, 5])
                        with three_columns[0]:
                            target_options = [50, 100, 1000, 10000]
                            target = st.selectbox("Select the target number of images labeled per class:", options=target_options, index=0)
                            fig = update_plot(count_df, target)
                            st.plotly_chart(fig, use_container_width=True)

                        with three_columns[2]:
                            st.markdown('#')
                            st.markdown('###')
                            st.markdown('###')
                            custom_colors = ['#0B5699', '#EDF8E6']
                            fig = px.bar(percent_df,
                                         x='class',
                                         y='% Images Labeled',
                                         color_discrete_sequence=custom_colors)
                            fig.update_layout(title_text='<i><b>Proportion of Validated Images</b></i>')

                            st.plotly_chart(fig, use_container_width=True)

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

                    three_columns = st.columns([.75,2.5,2.5])
                    with three_columns[0]:
                        st.subheader("Metrics")
                        st.metric("Accuracy:",
                                    f"{val_stats[0]*100:.2f} %",
                                    delta=f"{(val_stats[0] - model_stats[0])*100:.2f} %")
                        st.metric("Precision:",
                                    f"{val_stats[1]*100:.2f} %",
                                    delta=f"{(val_stats[1] - model_stats[1])*100:.2f} %")
                        st.metric("Recall:",
                                    f"{val_stats[2]*100:.2f} %",
                                    delta=f"{(val_stats[2] - model_stats[2])*100:.2f} %")

                    with three_columns[1]:
                        st.subheader("Confusion Matrix", 
                         help="""A confusion matrix is a tabular representation that
                         summarizes the effectiveness of a machine learning model
                         when tested against a dataset. It provides a visual breakdown
                         of correct and incorrect predictions made by the model.""")
                        c_report_test = ds.get_classification_report(
                                        validated_df,
                                        ['CONSENSUS', 'PRED_LABEL', None])
                        st.plotly_chart(ds.plot_confusion_matrix(validated_df,
                                            ['CONSENSUS', 'PRED_LABEL'],
                                            classes=c_report_test.index,
                                            normalize=True), use_container_width=True)
                    with three_columns[2]:
                        st.subheader("ROC Curve",
                         help="""An ROC (Receiver Operating Characteristic) curve,
                         illustrates how well a classification model performs across
                         various classification thresholds. It showcases two key
                         parameters: True Positive Rate and False Positive Rate.
                         The curve plots the relationship between TPR and FPR as the
                         classification threshold changes. Lowering the threshold
                         identifies more items as positive, leading to an increase in
                         both False Positives and True Positives.""")
                        st.plotly_chart(ds.plot_roc_curve(validated_df['CONSENSUS'],
                                        pd.DataFrame(validated_df['PROBS'].tolist()), 
                                        c_report_test.index.sort_values(ascending=True)),
                                        use_container_width=True)
                    two_columns = st.columns([4,3])
                    with two_columns[0]:
                        st.subheader("Model Performance: Precision, Recall, F1 Score", 
                         help="""Precision is the actual correct prediction divided by total
                        prediction made by model. Recall is the number of true positives
                        divided by the total number of true positives and false
                        negatives. F1 Score is the weighted average of precision and
                        recall.""")
                        c_report_test = c_report_test.sort_values(by=['f1-score'],
                                                                        ascending=False)
                        st.plotly_chart(ds.plot_precision_recall_f1(c_report_test),
                                    use_container_width=True)
                    with two_columns[1]:
                        st.subheader("Sunburst Plot", 
                         help="""This sunburst plot visualizes the CNN predicted labels
                         (inner circle) and the user verified labels (outer circle). """)                   
                        agg_df = validated_df.groupby(['PRED_LABEL', 'CONSENSUS']).size().reset_index(name='count')

                        fig = px.sunburst(agg_df, path=['PRED_LABEL', 'CONSENSUS'], values='count')
                        fig.update_traces(marker_colors=[
                                                px.colors.qualitative.Prism[c] for c in pd.factorize(fig.data[0].labels)[0]
                                            ],
                                            leaf_opacity=.8)
                        fig.update_layout(title_text='<i><b>Sunburst Plot</b></i>')
                        st.plotly_chart(fig, use_container_width=True)
