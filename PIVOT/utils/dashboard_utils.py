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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import streamlit as st

from sklearn.metrics import precision_score, recall_score, roc_auc_score, \
                            confusion_matrix, classification_report, roc_curve


def plot_confusion_matrix(cm_df, col_names, classes, normalize=False):
    """
    This function plots a confusion matrix.

    Args:
        - cm_df (DataFrame): A DataFrame containing the true and predicted classes.
        - col_names (list): List of column names that contain the true and predicted labels.
        - classes (list): List of class labels.
        - normalize (bool): Whether to normalize the matrix or not.

    Returns:
        - fig: A plotly figure.
    """

    # Create a confusion matrix
    con_matrix = confusion_matrix(cm_df[col_names[0]], cm_df[col_names[1]])

    x = classes
    y = classes

    # Normalize confusion matrix to scale values
    if normalize:
        con_matrix = np.around(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis], 2)
    z = con_matrix

    # Change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # Create the figure and add titles
    fig = ff.create_annotated_heatmap(z, x=list(x), y=list(y), annotation_text=z_text, colorscale='GnBu')
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # Adjust figure layout and add color bar
    fig.update_layout(margin=dict(t=50, l=200))
    fig['data'][0]['showscale'] = True

    return fig

@st.cache_data(ttl=500)
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

    return fig

@st.cache_data(ttl=500)
def get_classification_report(model_df, col_names, class_names = None):
    """
    This function gets the classification report and converts it in to a Pandas
    DataFrame.

    Args:
        - model_df (DataFrame): A Pandas DataFrame containing the true, predicted, and
                                names of the class labels.
        - col_names (list): List of column names that contain the true and predicted labels.
        - class_names (list) (optional): An ordered list of the class names.

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

@st.cache_data(ttl=500)
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
    """
    This function creates an ROC curve plot from input.

    Args:
        - true_label (pd.Series): A series of validated class labels.
        - prob_label (pd.Series): A series of predicted class labels.
        - classes (list): a list of class names.

    Returns:
        - fig: A plotly figure.
    """
    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(true_label, columns=classes)

    # Determine which classes do not exist in case of test_summary.
    # Classes that have yet to be classified will be filled with values of 0.
    if len(prob_label.columns) != len(classes):
        columns_not_in_list = [col for col in prob_label.columns if col not in classes]
        missing_columns_df = pd.DataFrame(0,
                                          index=y_onehot.index,
                                          columns=columns_not_in_list)
        y_onehot = pd.concat([y_onehot, missing_columns_df], axis=1)
        y_onehot = y_onehot.reindex(sorted(y_onehot.columns), axis=1)

    # Create an empty figure, and iteratively add new lines for each class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # Calculate the ROC and AUC score for each class and plot
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
        margin=dict(t=50, l=100)
    )
    return fig

@st.cache_data(ttl=500)
def target_plot(count_df, target):
    """
    Plots the number of images labeled for each class in comparison to a user-inputted target threshold.
    
    Parameters:
        count_df (pd.DataFrame): DataFrame with the number of images labeled per class.
        target (int): The target number of images to be labeled for each class.
    
    Returns:
        fig: a Plotly express bar chart.
    """
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

def class_proportion_plot(percent_df):
    """
    Creates a class proportion plot from a percentage df.
    """
    custom_colors = ['#1B7AB5']
    fig = px.bar(percent_df,
                 x='class',
                 y='% Images Labeled',
                 color_discrete_sequence=custom_colors)
    fig.update_layout(title_text='<i><b>Proportion of Validated Images</b></i>')

    return fig

@st.cache_data(ttl=500)
def plot_sunburst(agg_df):
    """
    Plots a sunburst plot from a aggregated prediction and label df.
    """
    fig = px.sunburst(agg_df, path=['PRED_LABEL', 'CONSENSUS'], values='count')
    fig.update_traces(marker_colors=[
        px.colors.qualitative.Prism[c] for c in pd.factorize(fig.data[0].labels)[0]],
                    leaf_opacity=.8,)

    return fig
