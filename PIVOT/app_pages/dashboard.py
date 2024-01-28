import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import itertools
import numpy as np

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report


def get_percent(row, c): 
    val = row[c]
    return float(val) / float(row['n_obs'])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.YlGnBu):
    """
    This function plots a confusion matrix.

    Args:
        cm (array): Confusion matrix.
        classes (list): List of class labels.
        normalize (bool): Whether to normalize the matrix or not.
        title (str): Plot title.
        cmap (matplotlib colormap): Colormap to be used for the plot.
    """
    fig, ax = plt.subplots() 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
        horizontalalignment="center",
        color="#04712f" if cm[i, j] > thresh else "grey")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig

def plot_precision_recall(class_labels, x_labels, precision, recall, spacing):
    fig, ax = plt.subplots() 
    plt.bar(x_labels - 0.2, precision, spacing, label="Precision", color='#1cb4ff')
    plt.bar(x_labels + 0.2, recall, spacing, label="Recall", color='#04712f')

    plt.xticks(x_labels, class_labels, rotation=45)
    plt.xlabel("Phytoplankton Classes") 
    plt.ylabel("Performance")
    plt.title("Model Performance: Precision and Recall") 
    plt.legend()

    return fig

def plot_f1_score(class_labels, f1_score):
    fig, ax = plt.subplots()
    plt.bar(class_labels, f1_score, color='#0d205f')

    plt.xticks(class_labels, rotation=45)
    plt.xlabel("Phytoplankton Classes") 
    plt.ylabel("F1 Score")
    plt.title("Model Performance: F1 Score") 

    return fig

def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

def main():

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)
    
    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Model Train Summary</h3>""",
                unsafe_allow_html=True)

    class_labels = ["Chloro", "Cilliate", "Crypto", "Diatom", "Dictyo", "Dino", "Eugleno", "Unident.", "Prymnesio", "null"]

    model_pred = pd.read_csv('data/model-summary-cnn-v1-b3.csv')
    model_acc = float(sum(model_pred['is_correct']))/float(len(model_pred))
    model_prec = precision_score(model_pred['true_label'],
                                 model_pred['pred_label'],
                                 average='weighted')
    model_recall = recall_score(model_pred['true_label'],
                                 model_pred['pred_label'],
                                 average='weighted')

    st.write("Accuracy: ", model_acc)
    st.write("Precision: ", model_prec)
    st.write("Recall: ", model_recall)

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
        if filtered_phyto[i] == False:
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

        cm = confusion_matrix(model_pred.true_label, model_pred.pred_label)
        cm_fig = plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Confusion Matrix')
        st.pyplot(cm_fig)

    with right_1:
        c_report = c_report.sort_values(by=['f1-score'], ascending=False)
        f1_plot = plot_f1_score(c_report['class_label'], c_report['f1-score'])
        st.pyplot(f1_plot)

    st.divider()

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Model Test Summary</h3>""",
                unsafe_allow_html=True)
    
    left_2, right_2 = st.columns(2)
    with left_2:
        test_model = st.selectbox(label='Select the model of interest:',
                                    options=('xxx', 'yyy', 'zzz'),
                                    format_func={'xxx': 'CNN ver. 1',
                                                'yyy': 'CNN ver. 2',
                                                'zzz': 'CNN ver. 3'}.__getitem__,
                                    index=None)

