"""
    Module for testing the dashboard utilities.
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import plotly.graph_objs as go

from unittest.mock import patch
import plotly.express as px
from utils import dashboard_utils as dau

class TestPlotConfusionMatrix(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'True_Labels': ['A', 'A', 'B', 'B', 'C', 'C'],
                'Predicted_Labels': ['A', 'B', 'B', 'C', 'C', 'C']}
        self.cm_df = pd.DataFrame(data)

        self.col_names = ['True_Labels', 'Predicted_Labels']
        self.classes = ['A', 'B', 'C']

    def test_plot_confusion_matrix(self):
        fig = dau.plot_confusion_matrix(self.cm_df, self.col_names, self.classes)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_confusion_matrix_normalized(self):
        fig = dau.plot_confusion_matrix(self.cm_df, self.col_names, self.classes, normalize=True)
        self.assertIsInstance(fig, go.Figure)

class TestPlotPrecisionRecallF1(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'class_label': ['A', 'B', 'C'],
                'precision': [0.85, 0.90, 0.75],
                'recall': [0.80, 0.88, 0.70],
                'f1-score': [0.82, 0.89, 0.72]}
        self.class_report = pd.DataFrame(data)

    def test_plot_precision_recall_f1(self):
        fig = dau.plot_precision_recall_f1(self.class_report)
        self.assertIsInstance(fig, go.Figure)

class TestGetClassificationReport(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.model_df = pd.DataFrame({
            'TrueLabel': [0, 1, 0, 1],
            'PredictedLabel': [0, 0, 1, 1]
        })
        self.col_names = ['TrueLabel', 'PredictedLabel']
        self.class_names = ['Class0', 'Class1']

    @patch('sklearn.metrics.classification_report')
    def test_valid_input(self, mock_class_report):
        # Mocking classification_report output
        mock_class_report.return_value = {
            '0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.67, 'support': 2},
            '1': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.67, 'support': 2},
            'accuracy': 0.5,
            'macro avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 4},
            'weighted avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 4}
        }

        report_df = dau.get_classification_report(self.model_df, self.col_names)
        # Assert DataFrame structure and content
        self.assertIn('precision', report_df.columns)
        self.assertIn('recall', report_df.columns)
        self.assertIn('f1-score', report_df.columns)
        self.assertIn('support', report_df.columns)
        self.assertIn('class_label', report_df.columns)
        # Additional assertions as needed

    def test_with_class_names(self):
        # Test with class_names provided
        report_df = dau.get_classification_report(self.model_df, self.col_names, self.class_names)
        # Assert class labels are assigned correctly
        self.assertTrue((report_df['class_label'] == self.class_names).all())

    def test_no_class_names(self):
        # Test with no class_names provided
        report_df = dau.get_classification_report(self.model_df, self.col_names)
        # Assert class labels are from index
        self.assertTrue((report_df['class_label'] == report_df.index).all())

class TestPlotSunburst(unittest.TestCase):

    def setUp(self):
        self.agg_df = pd.DataFrame({
            'PRED_LABEL': ['A', 'B', 'A', 'C'],
            'CONSENSUS': ['X', 'Y', 'Z', 'W'],
            'count': [10, 20, 30, 40]
        })

    def test_plot_sunburst(self):
        # Mocking px.sunburst
        px.sunburst = MagicMock()

        # Call the function
        dau.plot_sunburst(self.agg_df)

        # Assert that px.sunburst is called with the correct arguments
        px.sunburst.assert_called_once_with(
            self.agg_df, path=['PRED_LABEL', 'CONSENSUS'], values='count'
        )

    def test_plot_sunburst_return_type(self):
        fig = dau.plot_sunburst(self.agg_df)
        self.assertIsInstance(fig, type(px.sunburst()))

class TestClassProportionPlot(unittest.TestCase):

    def setUp(self):
        self.percent_df = pd.DataFrame({
            'class': ['A', 'B', 'C'],
            '% Images Labeled': [10, 20, 30]
        })

    def test_class_proportion_plot(self):
        # Mocking px.bar
        px.bar = MagicMock()

        # Call the function
        dau.class_proportion_plot(self.percent_df)

        # Assert that px.bar is called with the correct arguments
        px.bar.assert_called_once_with(
            self.percent_df,
            x='class',
            y='% Images Labeled',
            color_discrete_sequence=['#1B7AB5']
        )
class TestTargetPlot(unittest.TestCase):

    def setUp(self):
        self.count_df = pd.DataFrame({
            'class': ['A', 'B', 'C'],
            '# Images Labeled': [10, 20, 30]
        })
        self.target = 100

    def test_target_plot(self):
        # Mocking px.bar and go.Bar
        px.bar = MagicMock()
        go.Bar = MagicMock()

        # Call the function
        dau.target_plot(self.count_df, self.target)

        # Assert that px.bar is called with the correct arguments
        px.bar.assert_called_once_with(
            self.count_df,
            x='class',
            y='# Images Labeled',
            labels={'remaining': 'Remaining Images'},
            color='color',
            color_discrete_map={'red': 'red', 'green': 'green'}
        )

        # Assert that go.Bar is called with the correct arguments
        go.Bar.assert_called_once()

class TestPlotRocCurve(unittest.TestCase):

    def setUp(self):
        self.true_label = pd.Series([0, 1, 0, 1])
        self.prob_label = pd.DataFrame({
            'class_0': [0.2, 0.8, 0.3, 0.7],
            'class_1': [0.6, 0.3, 0.4, 0.9]
        })
        self.classes = ['class_0', 'class_1']

    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.roc_auc_score')
    def test_plot_roc_curve(self, mock_auc, mock_roc):
        dau.plot_roc_curve(self.true_label, self.prob_label, self.classes)
        # Assert that roc_curve and roc_auc_score are called
        mock_auc.called_once()
        mock_roc.called_once()

    def test_plot_roc_curve_layout(self):
        fig = dau.plot_roc_curve(self.true_label, self.prob_label, self.classes)
        self.assertEqual(fig.layout.xaxis.title.text, 'False Positive Rate')
        self.assertEqual(fig.layout.yaxis.title.text, 'True Positive Rate')
        self.assertEqual(fig.layout.margin.t, 50)
        self.assertEqual(fig.layout.margin.l, 100)

    def test_less_classes(self):
        self.classes = [0,1,2]
        self.prob_label = pd.DataFrame({
            0: [0.2, 0.8, 0.3, 0.7],
            1: [0.6, 0.3, 0.4, 0.9]
        })
        fig = dau.plot_roc_curve(self.true_label, self.prob_label, self.classes)
        self.assertEqual(fig.layout.xaxis.title.text, 'False Positive Rate')
        self.assertEqual(fig.layout.yaxis.title.text, 'True Positive Rate')
        self.assertEqual(fig.layout.margin.t, 50)
        self.assertEqual(fig.layout.margin.l, 100)

class TestGetAccPrecRecall(unittest.TestCase):

    def setUp(self):
        self.model_df = pd.DataFrame({
            'true_labels': [0, 1, 0, 1],
            'predicted_labels': [0, 1, 1, 1],
            'count_correct': [1, 1, 0, 1]
        })
        self.col_names = ['true_labels', 'predicted_labels', 'count_correct']

    def test_get_acc_prec_recall(self):
        apr = dau.get_acc_prec_recall(self.model_df, self.col_names)
        self.assertEqual(len(apr),3)
        self.assertEqual(apr[0], .5)

if __name__ == '__main__':
    unittest.main()