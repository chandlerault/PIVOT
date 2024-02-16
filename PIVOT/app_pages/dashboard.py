"""
Code that executes the contents of the Summary Metrics page
and is called by the main app.py script.

Functions:
    - plot_confusion_matrix: Plots the calculated confusion matrix with matplotlib.
    - plot_precision_recall: Plots a bar graph of each classes precision and recall.
    - plot_f1_score: Plots a bar graph of each classes F1 score.
    - get_classification_report: Calculates the classfication report using the models
                                 predicted and true labels.
    - main: Executes the Streamlit formatted HTML when called by app.py.
"""
import os
import itertools
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

from utils import sql_utils
from utils import app_utils

def plot_confusion_matrix(con_matrix, classes, normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.YlGnBu):
    """
    This function plots a confusion matrix.

    Args:
        - con_matrix (array): Confusion matrix.
        - classes (list): List of class labels.
        - normalize (bool): Whether to normalize the matrix or not.
        - title (str): Plot title.
        - cmap (matplotlib colormap): Colormap to be used for the plot.

    Returns:
        - fig: A matplotlib figure.
    """
    fig = plt.subplots()
    plt.imshow(con_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]

    thresh = con_matrix.max() / 2.

    for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
        plt.text(j, i, format(con_matrix[i, j], '.2f' if normalize else 'd'),
        horizontalalignment="center",
        color="#04712f" if con_matrix[i, j] > thresh else "grey")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig[0]

def plot_precision_recall(class_labels, x_labels, precision, recall, spacing):
    """
    This function plots a bar graph of a models precesion and recall by class.

    Args:
        - class_labels (list): The labels as strings of all model classes.
        - x_labels (int): Number of class labels.
        - precision (list): A list of precisions (as floats) for each class.
        - recall (list): A list of recalls (as floats) for each class.
        - spacing (float): Desired distance between bars.

    Returns:
        - fig: A matplotlib figure.
    """
    fig = plt.subplots()
    plt.bar(x_labels - 0.2, precision, spacing, label="Precision", color='#1cb4ff')
    plt.bar(x_labels + 0.2, recall, spacing, label="Recall", color='#04712f')

    plt.xticks(x_labels, class_labels, rotation=90)
    plt.xlabel("Phytoplankton Classes")
    plt.ylabel("Performance")
    plt.title("Model Performance: Precision and Recall")
    plt.legend()

    return fig[0]

def plot_f1_score(class_labels, f1_score):
    """
    This function plots a bar graph of a models f1 score by class.

    Args:
        - class_labels (list): The labels as strings of all model classes.
        - f1_score (list): A list of f1 scores (as floats) for each class.

    Returns:
        - fig: A matplotlib figure.
    """
    fig = plt.subplots()
    plt.bar(class_labels, f1_score, color='#0d205f')

    plt.xticks(class_labels, rotation=90)
    plt.xlabel("Phytoplankton Classes")
    plt.ylabel("F1 Score")
    plt.title("Model Performance: F1 Score")

    return fig[0]

def get_classification_report(y_test, y_pred):
    """
    This function gets the classfication report and converts it in to a Pandas
    DataFrame.

    Args:
        - y_test (list): A list of the actual phytoplankton labels.
        - y_pred (list): A list of the predicted phytoplankton labels.

    Returns:
        - df_classification_report: The classification report as a Pandas DataFrame.
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'],
                                                                    ascending=False)
    return df_classification_report

def main():
    """
    Executes the Streamlit formatted HTML displayed on the Dashboard subpage and
    displays summary statistics and graphs for the trained model and test model.
    Users can select from the different test models stored on the SQL Database.
    """

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Summary Metrics<br><br></h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    tab_1, tab_2 = st.tabs(['Train Summary', 'Test Summary'])
    with tab_1:
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
        model_acc = float(sum(model_pred['is_correct']))/float(len(model_pred))
        model_prec = precision_score(model_pred['true_label'],
                                    model_pred['pred_label'],
                                    average='weighted')
        model_recall = recall_score(model_pred['true_label'],
                                    model_pred['pred_label'],
                                    average='weighted')

        left, middle, right = st.columns(3)
        with left:
            st.metric("Accuracy:", f"{model_acc*100:.2f} %")
        with middle:
            st.metric("Precision:", f"{model_prec*100:.2f} %")
        with right:
            st.metric("Recall:", f"{model_recall*100:.2f} %")

        st.markdown("""<h1></h1>""", unsafe_allow_html=True)

        st.markdown("""<h6 style='text-align: left; color: black;'>
                    Filter Phytoplankton Subcategories</h6>""",
                    unsafe_allow_html=True)

        filtered_phyto = list(range(0,10,1))

        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            filtered_phyto[0] = st.checkbox(class_labels[0], value=1)
            filtered_phyto[5] = st.checkbox(class_labels[5], value=1)
        with col_2:
            filtered_phyto[1] = st.checkbox(class_labels[1], value=1)
            filtered_phyto[6] = st.checkbox(class_labels[6], value=1)
        with col_3:
            filtered_phyto[2] = st.checkbox(class_labels[2], value=1)
            filtered_phyto[7] = st.checkbox(class_labels[7], value=1)
        with col_4:
            filtered_phyto[3] = st.checkbox(class_labels[3], value=1)
            filtered_phyto[8] = st.checkbox(class_labels[8], value=1)
        with col_5:
            filtered_phyto[4] = st.checkbox(class_labels[4], value=1)
            filtered_phyto[9] = st.checkbox(class_labels[9], value=1)

        st.markdown("""<h1></h1>""", unsafe_allow_html=True)

        count = 0
        for i in range(0,len(filtered_phyto)):
            if filtered_phyto[i] is False:
                model_pred = model_pred[model_pred['true_label'] != i]
                model_pred = model_pred[model_pred['pred_label'] != i]
                i = i-count
                del class_labels[i]
                count = count + 1

        left_1, right_1 = st.columns(2)
        with left_1:
            c_report = get_classification_report(model_pred.true_label,model_pred.pred_label)
            c_report = c_report.drop(['weighted avg','accuracy','macro avg']).sort_index()
            c_report = c_report.assign(class_label=class_labels)
            prec_rec = plot_precision_recall(class_labels,
                                    np.arange(len(class_labels)),
                                    c_report.precision,
                                    c_report.recall,
                                    0.4)
            st.pyplot(prec_rec)

            st.markdown("""<h1></h1>""", unsafe_allow_html=True)

            con_max = confusion_matrix(model_pred.true_label, model_pred.pred_label)
            cm_fig = plot_confusion_matrix(con_max, classes=class_labels, normalize=True,
                                        title='Confusion Matrix')
            st.pyplot(cm_fig)

        with right_1:
            c_report = c_report.sort_values(by=['f1-score'], ascending=False)
            f1_plot = plot_f1_score(c_report['class_label'], c_report['f1-score'])
            st.pyplot(f1_plot)
    with tab_2:
        if os.stat("config/config.yaml").st_size == 0:
            st.error("""No database configuration found. Please update the database
                     configuration in the Settings page.""")
        else:
            model_list = app_utils.get_models()
            if not model_list:
                st.error("""Please ensure database configuration information is correct.
                         If so, restart app to ensure database configurations has been
                         saved with the following command.""")
                st.code("streamlit run app.py", language=None)
            else:
                left_2, right_2 = st.columns(2)
                with left_2:
                    model_list = app_utils.get_models()
                    model_dictionary = {}
                    model_name = []

                    for i in range(1,len(model_list)):
                        model_name.append(model_list[i]['m_id'])
                        model_dictionary[model_list[i]['m_id']] = model_list[i]['model_name']

                    selected_model_sum = st.selectbox(label='Select the model you wish to evaluate:',
                                                options=tuple(model_name),
                                                format_func=model_dictionary.__getitem__,
                                                index=None)
                with right_2:
                    pass

                st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                validated_df = sql_utils.get_test_set_df(selected_model_sum)
                if validated_df is not None:
                    validated_df['IS_CORRECT'] = (validated_df['PRED_LABEL'] == \
                                                validated_df['CONSENSUS']).astype(int)

                    val_acc = float(sum(validated_df['IS_CORRECT']))/float(len(validated_df))
                    val_prec = precision_score(validated_df['CONSENSUS'],
                                                validated_df['PRED_LABEL'],
                                                average='weighted')
                    val_recall = recall_score(validated_df['CONSENSUS'],
                                            validated_df['PRED_LABEL'],
                                            average='weighted')
                    
                    test_acc_diff = val_acc - model_acc
                    test_pre_diff = val_prec - model_prec
                    test_rec_diff = val_recall - model_recall
                    
                    left_4, middle_4, right_4 = st.columns(3)
                    with left_4:
                        st.metric("Accuracy:", f"{val_acc*100:.2f} %", delta=f"{test_acc_diff*100:.2f} %")
                    with middle_4:
                        st.metric("Precision:", f"{val_prec*100:.2f} %", delta=f"{test_pre_diff*100:.2f} %")
                    with right_4:
                        st.metric("Recall:", f"{val_recall*100:.2f} %", delta=f"{test_rec_diff*100:.2f} %")

                    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                    left_3, right_3 = st.columns(2)
                    with left_3:
                        c_report_test = get_classification_report(validated_df.CONSENSUS,
                                                                validated_df.PRED_LABEL)
                        c_report_test = c_report_test.drop(['weighted avg','accuracy','macro avg']) \
                            .sort_index()
                        c_report_test = c_report_test.assign(class_label=c_report_test.index)
                        prec_rec_test = plot_precision_recall(c_report_test.index,
                                                np.arange(len(c_report_test.index)),
                                                c_report_test.precision,
                                                c_report_test.recall,
                                                0.4)
                        st.pyplot(prec_rec_test)

                        st.markdown("""<h1></h1>""", unsafe_allow_html=True)

                        cm_test = confusion_matrix(validated_df.CONSENSUS, validated_df.PRED_LABEL)
                        cm_fig_test = plot_confusion_matrix(cm_test, classes=c_report_test.index,
                                                            normalize=True, title='Confusion Matrix')
                        st.pyplot(cm_fig_test)

                    with right_3:
                        c_report_test = c_report_test.sort_values(by=['f1-score'], ascending=False)
                        f1_plot_test = plot_f1_score(c_report_test['class_label'],
                                                    c_report_test['f1-score'])
                        st.pyplot(f1_plot_test)
