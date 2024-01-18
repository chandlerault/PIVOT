import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def main():

    st.markdown("""
            <h1 style='text-align: center; color: white; background-image: url(https://img.freepik.com/premium-photo/cute-colorful-abstract-background_480962-11756.jpg);
            padding-top: 70px''>
            Phytoplankton Image Validation Optimization Toolkit<br><br></h1>""",
            unsafe_allow_html=True)
    
    st.markdown("""<h1></h1>""", unsafe_allow_html=True)

    st.markdown("""<h3 style='text-align: left; color: black;'>
                Model Summary</h4>""",
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

    st.divider()

    left_1, right_1 = st.columns(2)
    with left_1:
        CM_fig = ConfusionMatrixDisplay.from_predictions(model_pred['true_label'],
                                                        model_pred['pred_label'],
                                                        display_labels=class_labels,
                                                        xticks_rotation=45)
        st.pyplot(CM_fig.figure_)

