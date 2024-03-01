"""
    Module for testing the app utilities.
"""
import unittest
import time
from unittest.mock import patch
import sys
from io import StringIO
from utils import app_utils as au
import numpy as np
import pandas as pd
class TestInsertLabel(unittest.TestCase):
    """
    Test case for the insert_label function.
    """
    @patch('utils.app_utils.data_utils.insert_data')
    @patch('utils.app_utils.sql_utils.update_scores')
    def test_insert_label_online(self,mock_sql, mock_insert):
        """
        Test insert_label function.
        """
        mock_insert.return_value.__enter__.return_value = None
        data = {
                'weight': [1, 3, 3, 4],
                'i_id': ['a', 'b', 'c', 'd'],
                'C': [True, False, True, False]
                }

        df = pd.DataFrame(data)
        au.insert_label(df)
        mock_insert.assert_called_once()
        self.assertEqual(mock_sql.call_count, len(df['weight'].unique()))

    @patch('utils.app_utils.data_utils.insert_data')
    def test_insert_label_error(self, mock_insert):
        """
        Test insert_label function for error.
        """
        mock_insert.return_value.__enter__.return_value = None
        data = {
                'A': [1, 2, 3, 4],
                'B': ['a', 'b', 'c', 'd'],
                'C': [True, False, True, False]
                }

        with self.assertRaises(TypeError):
            au.insert_label(data)

class TestAwaitConnection(unittest.TestCase):
    """
    Test case for the await_connection function.
    """
    @patch('utils.app_utils.data_utils.get_status')
    def test_get_status_true(self, mock_status):
        """
        Test await_connection for a connected db.
        """
        mock_status.return_value = True
        result = au.await_connection()
        self.assertTrue(result)

    @patch('utils.app_utils.data_utils.get_status')
    def test_get_status_false(self, mock_status):
        """
        Test await_connection for fail to connect.
        """
        mock_status.return_value = False

        start = time.time()
        result = au.await_connection(max_time=2, step=1)
        end = time.time()
        self.assertEqual(round(end-start), 2)
        self.assertFalse(result)
        self.assertEqual(mock_status.call_count, 2)

    @patch('utils.app_utils.data_utils.get_status')
    def test_get_status_alternate(self, mock_status):
        """
        Test await_connection for false then true.
        """
        mock_status.side_effect = [False, True]

        start = time.time()
        result = au.await_connection(max_time=3, step=1)
        end = time.time()
        self.assertEqual(round(end-start), 1)
        self.assertTrue(result)
        self.assertEqual(mock_status.call_count, 2)

    @patch('utils.app_utils.data_utils.get_status')
    def test_get_status_errors(self, mock_status):
        """
        Test await_connection for errors.
        """
        mock_status.side_effect = False
        with self.assertRaises(ValueError):
            au.await_connection(max_time=.3, step=1)
        with self.assertRaises(TypeError):
            au.await_connection(max_time=3, step=.1)
        with self.assertRaises(TypeError):
            au.await_connection(max_time=1.3, step=1)
        with self.assertRaises(ValueError):
            au.await_connection(max_time=-3, step=-10)
        with self.assertRaises(ValueError):
            au.await_connection(max_time=3, step=10)
        with self.assertRaises(ValueError):
            au.await_connection(max_time=3, step=-10)

class TestGetDissimilarity(unittest.TestCase):
    """
    Test case for the get_dissimilarity function.
    """
    @patch('utils.app_utils.data_utils.select_distinct')
    def test_get_dissimilarity(self, mock_select):
        """
        Test get_dissimilarity function.
        """
        mock_select.return_value = {'name': 'd', 'd_id':1}
        au.get_dissimilarities()
        mock_select.assert_called_once()

class TestGetModels(unittest.TestCase):
    """
    Test case for the get_models function.
    """
    @patch('utils.app_utils.data_utils.select_distinct')
    def test_get_models(self, mock_select):
        """
        Test get_models for a connected db.
        """
        mock_select.return_value = {'model_name': 'm', 'm_id':1}
        au.get_models()
        mock_select.assert_called_once()

class TestCreateUser(unittest.TestCase):
    """
    Test case for the create_user function.
    """
    @patch('utils.app_utils.data_utils.insert_data')
    def test_create_user(self, mock_insert):
        """
        Test create_user functions.
        """
        mock_insert.return_value = 1
        uid = au.create_user({'name': 'john doe', 'email': 'john@example.com'})
        mock_insert.assert_called_once()
        self.assertEqual(uid, 1)

class TestGetUser(unittest.TestCase):
    """
    Test case for the get_user function.
    """
    @patch('utils.app_utils.data_utils.select')
    def test_get_user(self, mock_select):
        """
        Test get_user functions.
        """
        mock_select.return_value = [{'name': 'john doe', 'email': 'john@example.com'}]
        user = au.get_user('john@example.com')
        mock_select.assert_called_once()
        self.assertDictEqual(user, {'name': 'john doe', 'email': 'john@example.com'})

    @patch('utils.app_utils.data_utils.select')
    def test_get_user_none(self, mock_select):
        """
        Test get_user function with no user.
        """
        mock_select.return_value = []
        uid = au.get_user('john@example.com')
        mock_select.assert_called_once()
        self.assertIsNone(uid)

        mock_select.return_value = None
        uid = au.get_user('john@example.com')
        self.assertIsNone(uid)

class TestGetImage(unittest.TestCase):
    """
    Test cases for the get_image function.
    """
    @patch('utils.app_utils.data_utils.get_blob_bytes')
    @patch('utils.app_utils.np.frombuffer')
    @patch('utils.app_utils.cv2.imdecode')
    def test_get_image_valid_png(self, mock_imdecode, mock_frombuffer, mock_get_blob_bytes):
        """
        Test a valid blob.
        """
        mock_get_blob_bytes.return_value = b'\x89PNG\r\n\x1a\n'
        mock_frombuffer.return_value = np.ones((105,105))
        mock_imdecode.return_value = np.ones((105,105))
        result = au.get_image("file_path")
        self.assertIsInstance(result, np.ndarray)


    @patch('utils.app_utils.data_utils.get_blob_bytes')
    def test_get_image_invalid_png(self, mock_get_blob_bytes):
        """
        Test an invalid blob.
        """
        mock_get_blob_bytes.return_value = b'\x00\x01\x02\x03'

        captured_output = StringIO()
        sys.stdout = captured_output

        result = au.get_image("file_path")

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertEqual(output.strip(), 'The blob does not appear to be a valid PNG image.')

        np.testing.assert_equal(result, np.zeros((128,128)))

if __name__ == '__main__':
    unittest.main()