"""
Module for testing the initiation of utils configuration.
"""

import unittest
from unittest.mock import patch, mock_open
import utils

class TestConfig(unittest.TestCase):
    """
    Test cases for util configuration
    """
    @patch('utils.yaml.load')
    @patch('utils.os.path.getmtime')
    @patch('builtins.open', new_callable=mock_open, read_data='Hello, world!')
    def test_changed_config(self, _, mock_mtime, mock_yaml_load):
        """
        Basic test case for a new config file.
        """
        mock_mtime.side_effect = [1, 2, 3,3,3]
        mock_yaml_load.side_effect = [None, {'data':1}]
        utils.load_config(file_path='test.txt', interval=.5)
        self.assertEqual(mock_yaml_load.call_count, 2)

    @patch('utils.yaml.load')
    @patch('utils.os.path.getmtime')
    @patch('builtins.open', new_callable=mock_open, read_data='Hello, world!')
    def test_changed_config_errors(self, _, mock_mtime, mock_yaml_load):
        """
        Basic test case for value errors
        """
        mock_mtime.side_effect = [1, 2, 3,3,3]
        mock_yaml_load.side_effect = [None, {'data':1}]
        with self.assertRaises(ValueError):
            utils.load_config(file_path='test.txt', interval=None)
if __name__ == '__main__':
    unittest.main()
