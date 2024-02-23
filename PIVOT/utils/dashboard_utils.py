"""
This file provides functions for the Summary Metrics page: Train and Test Summary tabs.

Functions:
    - plot_confusion_matrix: Plots the calculated confusion matrix with matplotlib.
    - plot_precision_recall: Plots a bar graph of each classes precision and recall.
    - plot_f1_score: Plots a bar graph of each classes F1 score.
    - get_classification_report: Calculates the classification report using the models
                                 predicted and true labels.
    - get_acc_prec_recall: Calculates the accuracy, precision, and recall of the output
                           DataFrame of a classification model.
"""
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

def plot_confusion_matrix(cm_df, col_names, classes, normalize=False, cmap=plt.cm.YlGnBu):
    """
    This function plots a confusion matrix.

    Args:
        - cm_df (DataFrame): A DataFrame containing the true and predicted classes.
        - col_names (list): List of column names that contain the true and predicted labels.
        - classes (list): List of class labels.
        - normalize (bool): Whether to normalize the matrix or not.
        - title (str): Plot title.
        - cmap (matplotlib colormap): Colormap to be used for the plot.

    Returns:
        - fig: A matplotlib figure.
    """
    con_matrix = confusion_matrix(cm_df[col_names[0]], cm_df[col_names[1]])

    fig = plt.subplots()
    plt.imshow(con_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
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

def plot_precision_recall(class_report):
    """
    This function plots a bar graph of a models precision and recall by class.

    Args:
        - class_report (DataFrame): A classification report containing the precision, recall,
                                    and names of each class.

    Returns:
        - fig: A plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_report['class_label'],
                         y=class_report['precision'],
                         name='Precision',
                         marker_color='#1cb4ff'))
    fig.add_trace(go.Bar(x=class_report['class_label'],
                         y=class_report['recall'],
                         name='Recall',
                         marker_color='#04712f'))
    fig.update_layout(title_text='Model Performance: Precision and Recall')

    return fig

def plot_f1_score(class_report):
    """
    This function plots a bar graph of a models f1 score by class.

    Args:
        - class_labels (list): The labels as strings of all model classes.
        - f1_score (list): A list of f1 scores (as floats) for each class.

    Returns:
        - fig: A matplotlib figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_report['class_label'], 
                         y=class_report['f1-score'],
                         marker_color='#0d205f'))
    fig.update_layout(title_text='Model Performance: F1 Score')

    return fig

def get_classification_report(model_df, col_names, class_names = None):
    """
    This function gets the classification report and converts it in to a Pandas
    DataFrame.

    Args:
        - model_df (DataFrame): A Pandas DataFrame containing the true, predicted, and
                                names of the class labels.
        - col_names (list): A list of the column names

    Returns:
        - c_report: The filtered classification report as a Pandas DataFrame.
    """
    report = classification_report(model_df[col_names[0]],
                                   model_df[col_names[1]],
                                   output_dict=True)

    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'],
                                                                    ascending=False)
    c_report = df_classification_report.drop(['weighted avg','accuracy','macro avg']) \
                                        .sort_index()
    if class_names:
        c_report = c_report.assign(class_label=class_names)
    else:
        c_report = c_report.assign(class_label=c_report.index)
    c_report = c_report.sort_values(by='precision', ascending=False)

    return c_report

def get_acc_prec_recall(model_df, col_names):
    """
    This function calculates the accuracy, precision, and recall of a classified model.

    Args:
        - model_df (DataFrame): A Pandas DataFrame containing the true, predicted, and
                                count of correct labels.
        - col_names (list): A list of the column names that contain the true, predicted,
                            and count of correct labels.

    Returns:
        - (accuracy, precision, recall): The accuracy, precision and recall as a tuple.
    """
    accuracy = float(sum(model_df[col_names[0]]))/float(len(model_df))
    precision = precision_score(model_df[col_names[1]],
                                model_df[col_names[2]],
                                average='weighted')
    recall = recall_score(model_df[col_names[1]],
                          model_df[col_names[2]],
                          average='weighted')
    return (accuracy, precision, recall)
