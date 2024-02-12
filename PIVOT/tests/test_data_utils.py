"""
    Module for testing the data utilities.
"""
import unittest
from unittest.mock import patch, MagicMock
from utils import data_utils as du
from utils import CONFIG


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

        result = du.get_status()

        # Assertions
        self.assertFalse(result)

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
        # Call the function with invalid type
        with self.assertRaises(TypeError):
            du.get_blob_bytes(123)  # Not a string

if __name__ == '__main__':
    unittest.main()
   