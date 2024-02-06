"""
    Module for testing the data manager.
"""
import unittest
from collections import OrderedDict
from unittest.mock import patch, MagicMock
import pandas as pd
from utils import data_utils as du
from utils import CONFIG


# class TestDataUtils(unittest.TestCase):
#     """
#     Test class for data utils module
#     """
#     def setUp(self):
#         pass
class TestGetStatus(unittest.TestCase):

    @patch('utils.data_utils.pymssql.connect')
    def test_get_status_online(self, mock_connect):
        # Return ONLINE 
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ['ONLINE']
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        result = du.get_status()
        database = CONFIG['database']

        # Assertions
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once_with("SELECT state_desc FROM sys.databases WHERE name = %s", (database,))

    @patch('utils.data_utils.pymssql.connect')
    def test_get_status_offline(self, mock_connect):
        # Return offline status
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
        # Setup mock to raise an exception
        mock_connect.side_effect = Exception("Connection error")

        result = du.get_status()

        # Assertions
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
   