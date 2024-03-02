"""
    Module for testing the data utilities.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import sys
from io import StringIO
from utils import data_utils as du
from utils import CONFIG
import pymssql
import numpy as np



class TestGetStatus(unittest.TestCase):
    """
    Test case for the get_status function.
    """
    @patch('utils.data_utils.pymssql.connect')
    def test_get_status_online(self, mock_connect):
        """
        Test get_status function when the database is online.
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ['ONLINE']
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        result = du.get_status()
        database = CONFIG['database']

        # Assertions
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once_with(
            "SELECT state_desc FROM sys.databases WHERE name = %s", (database,))

    @patch('utils.data_utils.pymssql.connect')
    def test_get_status_offline(self, mock_connect):
        """
        Test get_status function when the database is offline.
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ['OFFLINE']
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        result = du.get_status()

        # Assertions
        self.assertFalse(result)

    @patch('utils.data_utils.pymssql.connect')
    def test_get_status_exception(self, mock_connect):
        """
        Test get_status function when an exception is raised during database connection.
        """
        mock_connect.side_effect = Exception("Connection error")
        with self.assertRaises(Exception):
            du.get_status()


class TestGetBlobBytes(unittest.TestCase):
    """
    Test cases for the get_blob_bytes function.
    """
    @patch('utils.data_utils.BlobServiceClient')
    def test_get_blob_bytes_success(self, mock_blob_service_client):
        """
        Test get_blob_bytes function when the blob exists.
        """
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = True
        mock_blob_data = MagicMock()
        mock_blob_data.readall.return_value = b'blob data'
        mock_blob_client.download_blob.return_value = mock_blob_data
        mock_blob_service_client.from_connection_string.return_value.get_blob_client.return_value = mock_blob_client #pylint: disable=line-too-long

        result = du.get_blob_bytes('path/to/blob')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result, b'blob data')

    @patch('utils.data_utils.BlobServiceClient')
    def test_get_blob_bytes_blob_not_exist(self, mock_blob_service_client):
        """
        Test get_blob_bytes function when the blob does not exist.
        """
        # Setup mock BlobClient to simulate blob not existing
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = False
        mock_blob_service_client.from_connection_string.return_value.get_blob_client.return_value = mock_blob_client #pylint: disable=line-too-long

        result = du.get_blob_bytes('path/to/nonexistent/blob')

        # Assertions
        self.assertIsNone(result)

    @patch('utils.data_utils.BlobServiceClient')
    def test_get_blob_bytes_invalid_type(self, _):
        """
        Test get_blob_bytes function with invalid input type.
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        self.assertIsNone(du.get_blob_bytes(123))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('Error'))

class TestInsertData(unittest.TestCase):
    """
    Test case for the insert_data function.
    """
    @patch('utils.data_utils.pymssql.connect')
    def test_insert_list_data(self, mock_connect):
        """
        Test insert_data function with a list of dictionaries as input.
        """
        mock_cursor = MagicMock()
        mock_cursor.executemany.return_value.__enter__.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        data = [{'column1': 'value1', 'column2': 'value2'}, {'column1': 'value3', 'column2': 'value4'}]
        table_name = 'your_table'

        du.insert_data(table_name, data)

        mock_cursor.executemany.assert_called_once_with(
            'INSERT INTO your_table (column1, column2) VALUES (%(column1)s, %(column2)s)',
            [{'column1': 'value1', 'column2': 'value2'}, {'column1': 'value3', 'column2': 'value4'}])

    @patch('utils.data_utils.pymssql.connect')
    def test_insert_dict(self, mock_connect):
        """
        Test get_status function when the database is online.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        data = {'column1': 'value1', 'column2': 'value2'}
        table_name = 'your_table'

        du.insert_data(table_name, data)

        self.assertEqual(mock_cursor.execute.call_count, 2)

        expected_call = call('INSERT INTO your_table (column1, column2) VALUES (%(column1)s, %(column2)s)',
                             {'column1': 'value1', 'column2': 'value2'})
        self.assertIn(expected_call, mock_cursor.execute.call_args_list)

    @patch('utils.data_utils.pymssql.connect')
    def test_insert_data_exception_database(self, mock_connect):
        """
        Test insert_data function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymssql.DatabaseError("Database error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        table_name = 'test_table'
        data = {'column1': 'value1', 'column2': 'value2'}

        # Call the function and assert that it raises a DatabaseError
        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertIsNone(du.insert_data(table_name, data))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('DatabaseError'))

    @patch('utils.data_utils.pymssql.connect')
    def test_insert_data_exception_interface(self, mock_connect):
        """
        Test insert_data function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymssql.InterfaceError("Interface error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        table_name = 'test_table'
        data = {'column1': 'value1', 'column2': 'value2'}

        # Call the function and assert that it raises a DatabaseError
        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertIsNone(du.insert_data(table_name, data))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('InterfaceError'))

class TestSelect(unittest.TestCase):
    """
    Test case for the select function.
    """
    @patch('utils.data_utils.pymssql.connect')
    def test_select_data(self, mock_connect):
        """
        Test select function with a list of dictionaries as input.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.return_value = [(1,1),(2,2),(3,3)]
        mock_cursor.description = [['c1'], ['c2']]
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        conditions = {'c1': 'v1', 'c2': 'v2'}
        table_name = 'your_table'

        result = du.select(table_name, conditions)
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT * FROM your_table WHERE c1 = 'v1' AND c2 = 'v2'")
        expected_result = [{'c1':1,'c2':1}, {'c1':2,'c2':2}, {'c1':3,'c2':3}]
        self.assertListEqual(result, expected_result)

    @patch('utils.data_utils.pymssql.connect')
    def test_select_data_exception_database(self, mock_connect):
        """
        Test select function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.side_effect = pymssql.DatabaseError("Database error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        conditions = {'c1': 'v1', 'c2': 'v2'}
        table_name = 'your_table'


        # Call the function and assert that it raises a DatabaseError
        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertListEqual([],du.select(table_name, conditions))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('DatabaseError'))

    @patch('utils.data_utils.pymssql.connect')
    def test_select_data_exception_interface(self, mock_connect):
        """
        Test select function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = pymssql.InterfaceError("Interface error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        conditions = {'c1': 'v1', 'c2': 'v2'}
        table_name = 'your_table'


        # Call the function and assert that it raises a DatabaseError
        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertListEqual([],du.select(table_name, conditions))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('InterfaceError'))

class TestSelectDistinct(unittest.TestCase):
    """
    Test case for the select_distinct function.
    """
    @patch('utils.data_utils.pymssql.connect')
    def test_select_distinct(self, mock_connect):
        """
        Test select function with a list of dictionaries as input.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.return_value = [(1,1),(2,2),(3,3)]
        mock_cursor.description = [['c1'], ['c2']]
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        columns = ('c1','c2')
        table_name = 'your_table'

        result = du.select_distinct(table_name, columns)
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT DISTINCT c1, c2 FROM your_table")
        expected_result = [{'c1':1,'c2':1}, {'c1':2,'c2':2}, {'c1':3,'c2':3}]
        self.assertListEqual(result, expected_result)

    @patch('utils.data_utils.pymssql.connect')
    def test_select_distinct_exception_database(self, mock_connect):
        """
        Test select function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.side_effect = pymssql.DatabaseError("Database error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        columns = ('c1','c2')
        table_name = 'your_table'


        # Call the function and assert that it raises a DatabaseError
        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertListEqual([],du.select_distinct(table_name, columns))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('DatabaseError'))

    @patch('utils.data_utils.pymssql.connect')
    def test_select_data_exception_interface(self, mock_connect):
        """
        Test select function when a pymssql.DatabaseError is raised during database connection or execution.
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = pymssql.InterfaceError("Interface error")
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        columns = ('c1','c2')
        table_name = 'your_table'

        captured_output = StringIO()
        sys.stdout = captured_output

        self.assertListEqual([],du.select_distinct(table_name, columns))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertTrue(output.strip().startswith('InterfaceError'))

class TestPreprocessInput(unittest.TestCase):
    """
    Test case for the preprocess_input function.
    """
    def test_proprocess_input(self):
        """
        Test preprocess_input function with a image the same size as the reshaped image.
        """
        image = np.zeros((128, 128))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

        image = np.zeros((128, 128, 1))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

    def test_proprocess_input_large(self):
        """
        Test preprocess_input function with a larger image than target.
        """
        image = np.zeros((346, 128))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

        image = np.zeros((346, 650))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))
        
        image = np.zeros((100, 650))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

        image = np.zeros((200, 100))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

    def test_proprocess_input_small(self):
        """
        Test preprocess_input function with a smaller image than target.
        """
        image = np.zeros((64, 64))
        processed_image = du.preprocess_input(image)
        self.assertTupleEqual(processed_image.shape, (128,128, 1))

if __name__ == '__main__':
    unittest.main()
   