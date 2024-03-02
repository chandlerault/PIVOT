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
import streamlit as st

from sklearn.metrics import precision_score, recall_score, roc_auc_score, \
                            confusion_matrix, classification_report, roc_curve, auc


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

    x = classes
    y = classes

    if normalize:
        con_matrix = np.around(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis], 2)
    z = con_matrix

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=list(x), y=list(y), annotation_text=z_text, colorscale='GnBu')

    # add title
    fig.update_layout(title_text='<i><b>Confusion Matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig

def plot_precision_recall_f1(class_report):
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
                         marker_color='#094789'))
    fig.add_trace(go.Bar(x=class_report['class_label'],
                         y=class_report['recall'],
                         name='Recall',
                         marker_color='#9cd9b9'))
    fig.add_trace(go.Bar(x=class_report['class_label'], 
                         y=class_report['f1-score'],
                         name='F1 Score',
                         marker_color='#4cb1d2'))
    fig.update_layout(title_text='<i><b>Model Performance: Precision, Recall, and F1 Score</b></i>')

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

def plot_roc_curve(true_label, prob_label, classes):
            
    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(true_label, columns=classes)

    if len(prob_label.columns) != len(classes):
        columns_not_in_list = [col for col in prob_label.columns if col not in classes]
        missing_columns_df = pd.DataFrame(0,
                                          index=y_onehot.index,
                                          columns=columns_not_in_list)
        y_onehot = pd.concat([y_onehot, missing_columns_df], axis=1)
        y_onehot = y_onehot.reindex(sorted(y_onehot.columns), axis=1)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    count = 0
    for i in range(prob_label.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = prob_label.iloc[:, i]

        if sum(y_true.values) != 0:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{classes[count]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
            count = count +1

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title_text='<i><b>ROC Curve</b></i>',
        margin=dict(t=50, l=100)
        #yaxis=dict(scaleanchor="x", scaleratio=1),
        #xaxis=dict(constrain='domain'),
        #width=700, height=500
    )
    return fig
