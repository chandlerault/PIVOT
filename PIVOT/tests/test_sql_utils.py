"""
    Module for testing SQL utilities.
"""
import unittest
from unittest.mock import patch, MagicMock, call, mock_open

import sys
from collections import OrderedDict
from io import StringIO
from utils import sql_utils as su
from utils import CONFIG
import pymssql
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd
import warnings 
from collections.abc import Iterable




class TestGetImagesToMetrize(unittest.TestCase):
    """
    Test cases for getting images to metrize
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_get_images_to_metrize(self, mock_map_probs, mock_stored_procedure):
        """
        Basic test case.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': ['25', '30'],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)

        mock_stored_procedure.return_value = df
        mock_map_probs.return_value = pd.Series([{'something': 1,'something2': 1},
                                                 {'something': 1,'something2': 1}])

        result =  su.get_images_to_metrize(model_id=1, dissimilarity_id=1, server_args=su.get_server_arguments())
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': [{'something': 1,'something2': 1},{'something': 1,'something2': 1}],
            'City': ['New York', 'Los Angeles']
        }
        expected = pd.DataFrame(data)

        assert_frame_equal(result,expected)

class TestImagesToPredict(unittest.TestCase):
    """
    Test cases for getting images to predict.
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    def test_get_images_to_metrize(self,mock_stored_procedure):
        """
        Basic test case.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': ['25', '30'],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)

        mock_stored_procedure.return_value = df
        model_id = 1
        su.get_images_to_predict(model_id=model_id)

        mock_stored_procedure.assert_called_once_with(sp="GENERATE_IMAGES_TO_PREDICT",
                                                      args=OrderedDict([("MODEL_ID", model_id)]),
                                                      server_args={})


class TestGetTestSetDf(unittest.TestCase):
    """
    Test cases for getting test set
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_basic_functionality(self, mock_class_map, mock_stored_procedure):
        """
        Basic test case.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': [{0:.25, 1:.75}, {0:.85, 1:.15}],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        sp_name = 'MODEL_EVALUATION_MAX_CONSENSUS_FILTERING'
        model_id = 1
        minimum_percent = .5
        mock_class_map.return_value = pd.Series([{"c1":.25, "c2":.75}, {"c1":.85, "c2":.15}])
        su.get_test_set_df(model_id=model_id,
                    minimum_percent=minimum_percent,
                    sp_name = sp_name,
                    server_args={})

        mock_stored_procedure.assert_called_once()

    @patch('utils.sql_utils.execute_stored_procedure')
    def test_invalid_arguments(self, mock_stored_procedure):
        """
        Tests with wrong arguments.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': ['25', '30'],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        sp_name = 'MODEL_EVALUATION_MAX_CONSENSUS_FILTERING'
        model_id = 1
        minimum_percent = .5

        with self.assertRaises(ValueError):
            su.get_test_set_df(model_id=model_id,
                        minimum_percent=minimum_percent,
                        sp_name = 'sp_name',
                        server_args={})
        with self.assertRaises(ValueError):
            su.get_test_set_df(model_id=model_id,
                        minimum_percent=-1,
                        sp_name = sp_name,
                        server_args={})

class TestGetLabelRankDf(unittest.TestCase):
    """
    Test cases for getting label ranked df.
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_basic_functionality(self, mock_map, mock_stored_procedure):
        """
        Basic test case.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': ['25', '30'],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        mock_map.return_value = 'test'

        result = su.get_label_rank_df(model_id=1,
                      dissimilarity_id=1,
                      batch_size= 100,
                      random_ratio= 0.5)
        self.assertEqual(result.shape[0], 2*df.shape[0])

    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_value_error(self, mock_map, mock_stored_procedure):
        """
        Basic error cases.
        """
        data = {
            'Name': ['Alice', 'Bob'],
            'PROBS': ['25', '30'],
            'City': ['New York', 'Los Angeles']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        mock_map.return_value = 'test'


        with self.assertRaises(ValueError):
            su.get_label_rank_df(model_id=1,
                        dissimilarity_id=1,
                        batch_size= 100,
                        random_ratio= 1.5)

        with self.assertRaises(ValueError):
            su.get_label_rank_df(model_id=1,
                        dissimilarity_id=1,
                        batch_size= 100,
                        random_ratio= -0.5)

        with self.assertRaises(ValueError):
            su.get_label_rank_df(model_id=1,
                        dissimilarity_id=1,
                        batch_size= -100,
                        random_ratio= 0.5)

        mock_stored_procedure.side_effect = [None, None]
        with self.assertRaises(ValueError):
            with self.assertWarns(UserWarning):
                su.get_label_rank_df(model_id=1,
                            dissimilarity_id=1,
                            batch_size= 100,
                            random_ratio= .1)
                
        mock_stored_procedure.side_effect = [None, None]
        with self.assertRaises(ValueError):
            with self.assertWarns(UserWarning):
                su.get_label_rank_df(model_id=1,
                            dissimilarity_id=1,
                            batch_size= 100,
                            random_ratio= 1)
class TestGetTrainDf(unittest.TestCase):
    """
    Test cases for getting train df.
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_basic_functionality(self, mock_class_map, mock_stored_procedure):
        """
        Basic test case.
        """
        data = {
            'ALL_LABELS': ['Alice', 'Bob'],
            'PROBS': [{0:.25, 1:.75}, {0:.85, 1:.15}],
            'Labels': ['.3, .4', '1.0, 3.4'],
            'PercentConsensus':['.3, .5','.1, .9']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        model_id = 1
        dissimilarity_id = 1
        mock_class_map.return_value = pd.Series([{"c1":.25, "c2":.75}, {"c1":.85, "c2":.15}])
        su.get_train_df(model_id=model_id,
                        dissimilarity_id=dissimilarity_id,
                    all_classes=['minimum_percent'],
                    train_size=100)

        mock_stored_procedure.assert_called_once()

    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.map_probs_column')
    def test_value_errors(self, mock_class_map, mock_stored_procedure):
        """
        Basic for error throwing.
        """
        data = {
            'ALL_LABELS': ['Alice', 'Bob'],
            'PROBS': [{0:.25, 1:.75}, {0:.85, 1:.15}],
            'Labels': ['.3, .4', '1.0, 3.4'],
            'PercentConsensus':['.3, .5','.1, .9']
        }

        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        model_id = 1
        dissimilarity_id = 1
        mock_class_map.return_value = pd.Series([{"c1":.25, "c2":.75}, {"c1":.85, "c2":.15}])
        with self.assertRaises(ValueError):
            su.get_train_df(model_id=model_id,
                        dissimilarity_id=dissimilarity_id,
                        all_classes=['minimum_percent'],
                        train_size=0)

        with self.assertRaises(ValueError):
            su.get_train_df(model_id=model_id,
                        dissimilarity_id=dissimilarity_id,
                        all_classes=['minimum_percent'],
                        train_ids=['1',2],
                        train_size=0)

        with self.assertRaises(ValueError):
            su.get_train_df(model_id=model_id,
                        dissimilarity_id=dissimilarity_id,
                        all_classes=['minimum_percent'],
                        train_ids=1,
                        train_size=0)


class TestValidateArgs(unittest.TestCase):
    """
    Test cases for validating arguments of stored processes.
    """
    def test_warning(self):
        """
        Simple test to see if warning occurs.
        """
        with self.assertWarns(Warning):
            su.validate_args('fake', OrderedDict())

    def test_value_error(self):
        """
        Simple test to see error is raised.
        """
        with self.assertRaises(ValueError):
            su.validate_args('MODEL_EVALUATION_NON_TEST', OrderedDict())
        with self.assertRaises(ValueError):
            su.validate_args('MODEL_EVALUATION_NON_TEST', OrderedDict([('MODEL_ID', 1),
                                                                       ('MINIMUM_PERCENT', 'two')]))

    # def test_works(self):
    #     """
    #     Simple test to see if no error occurs.
    #     """
    #     result = su.validate_args('MODEL_EVALUATION_NON_TEST', OrderedDict([('MODEL_ID', 1),
    #                                                                         ('MINIMUM_PERCENT', 1.2)]))
    #     self.assertIsNone(result)

class TestGetServerArguments(unittest.TestCase):
    """
    Test case for the preprocess_input function.
    """
    def test_get_server_args(self):
        """
        Simple test to see if arguments are correctly taken from config file.
        """
        results = su.get_server_arguments()
        self.assertIsInstance(results, tuple)
        self.assertTupleEqual(results, (CONFIG['server'],CONFIG['database'],
                                        CONFIG['db_user'],CONFIG['db_password']))
    def test_get_custom_args(self):
        """
        Test custom server args.
        """
        server_args = {'database': 'test_db'}
        results = su.get_server_arguments(server_args=server_args)
        self.assertIsInstance(results, tuple)
        self.assertTupleEqual(results, (CONFIG['server'],'test_db',
                                        CONFIG['db_user'],CONFIG['db_password']))
    def test_bad_args(self):
        """
        Test passing bad arguments.
        """
        server_args = ['test_db']
        with self.assertRaises(AttributeError):
            su.get_server_arguments(server_args=server_args)

class TestExecuteStoredProcedure(unittest.TestCase):
    """
    Test cases for executing a stored procedure
    """
    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.validate_args')
    @patch('utils.sql_utils.get_server_arguments')
    @patch('utils.sql_utils.generate_arg_strings')
    def test_basic_functionality(self, mock_generate, mock_server_args, _, mock_connect):
        """
        Tests basic functionality
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.callproc.return_value = None
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_cursor.fetchall.return_value = [[1,2],[3,4], [1,4]]

        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_server_args.return_value = ('server', 'database', 'user', 'password' )
        mock_generate.return_value = 'test arg string'

        sp_name = "sp"
        test_arguments = {'test1': "arg1"}
        results = su.execute_stored_procedure(sp_name, test_arguments)

        self.assertIsInstance(results, pd.DataFrame)
        mock_connection.cursor.assert_called_once()
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.callproc.assert_called_once_with(sp_name, ("arg1",))

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.validate_args')
    @patch('utils.sql_utils.get_server_arguments')
    @patch('utils.sql_utils.generate_arg_strings')
    def test_arg_workaround(self, mock_generate, mock_server_args, _, mock_connect):
        """
        Test for argument workaround with long argument.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.callproc.return_value = None
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_cursor.fetchall.return_value = [[1,2],[3,4], [1,4]]

        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_server_args.return_value = ('server', 'database', 'user', 'password' )
        mock_generate.return_value = 'test arg string'

        sp_name = "sp"
        long_string = "abcdefghi" * 1000
        test_arguments = {'test1': long_string}
        with self.assertWarns(Warning):
            results = su.execute_stored_procedure(sp_name, test_arguments)

        self.assertIsInstance(results, pd.DataFrame)
        mock_connection.cursor.assert_called_once()
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.execute.assert_called_once_with(f"EXECUTE {sp_name} test arg string")

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.validate_args')
    @patch('utils.sql_utils.get_server_arguments')
    @patch('utils.sql_utils.generate_arg_strings')
    def test_error(self, mock_generate, mock_server_args, _, mock_connect):
        """
        Tests error throwing.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_cursor.callproc.return_value = None
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_cursor.rowcount = -1
        mock_cursor.fetchall.side_effect = pymssql.OperationalError("executed statement has no resultset")

        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_server_args.return_value = ('server', 'database', 'user', 'password' )
        mock_generate.return_value = 'test arg string'

        sp_name = "sp"
        short_string = "abcdefghi"
        test_arguments = {'test1': short_string}
        with self.assertWarns(Warning):
            results = su.execute_stored_procedure(sp_name, test_arguments)

        self.assertIsNone(results)
        
class TestLoadFileFromSQL(unittest.TestCase):
    """
    Test cases for loading a sql file.
    """
    @patch('builtins.open', new_callable=mock_open, read_data='SELECT * FROM table;')
    def test_successful(self, _):
        """
        Tests to see successful read.
        """
        sql_script = su.load_file_from_sql('file_path')

        self.assertEqual(sql_script, 'SELECT * FROM table;')

    @patch('builtins.open', new_callable=mock_open, read_data=None)
    def test_file_empty(self, _):
        """
        Test when file is empty
        """
        sql_script = su.load_file_from_sql('file_path')

        self.assertEqual(sql_script,'')

class TestCreateAlterStoredProcedure(unittest.TestCase):
    """
    Test cases for creating and altering a stored procedure.
    """
    @patch('utils.sql_utils.pymssql.connect')
    def test_basic_functionality(self, mock_connect):
        """
        Tests a simple use case
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        su.create_alter_stored_procedure(sp_name='GENERATE_IMAGES_TO_PREDICT')

        mock_cursor.execute.assert_called_once()

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.load_file_from_sql')
    def test_from_file(self, mock_load, mock_connect):
        """
        Tests loadinbg from file
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_load.return_value = None
        file_name = "test_file.sql"

        su.create_alter_stored_procedure(sp_name='GENERATE_IMAGES_TO_PREDICT', file_path=file_name)

        mock_load.assert_called_once_with(file_name)

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.load_file_from_sql')
    def test_fail_file_load(self, mock_load, mock_connect):
        """
        Tests loading a file that doesn't exist.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_load.side_effect = FileNotFoundError

        file_name = "test_file.sql"

        with self.assertRaises(FileNotFoundError):
            su.create_alter_stored_procedure(sp_name='GENERATE_IMAGES_TO_PREDICT', file_path=file_name)

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.load_file_from_sql')
    def test_pymssql_error(self, mock_load, mock_connect):
        """
        Tests loading a file that doesn't exist.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymssql.Error
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_load.return_value = 'select * from test'

        file_name = "test_file.sql"

        with self.assertRaises(pymssql.Error):
            su.create_alter_stored_procedure(sp_name='GENERATE_IMAGES_TO_PREDICT', file_path=file_name)

    @patch('utils.sql_utils.pymssql.connect')
    @patch('utils.sql_utils.load_file_from_sql')
    def test_warning(self, mock_load, mock_connect):
        """
        Tests loading a file that doesn't exist.
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = None
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_load.return_value = 'select * from test'

        file_name = "test_file.sql"
        sp_name = 'non-existent'

        with self.assertWarns(Warning):
            su.create_alter_stored_procedure(sp_name=sp_name, file_path=file_name)

class TestRunSQLQuery(unittest.TestCase):
    """
    Test cases for running sql queries.
    """
    @patch('utils.sql_utils.pymssql.connect')
    def test_basic_functionality(self, mock_connect):
        """
        Tests a simple SQL query execution
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.return_value = [[1,2],[3,4], [1,4]]
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        query = "Select * from table"

        results = su.run_sql_query(query=query)

        mock_cursor.execute.assert_called_once_with(query)
        mock_cursor.fetchall.assert_called_once()

        self.assertIsInstance(results, pd.DataFrame)

        assert_frame_equal(results, pd.DataFrame([[1,2],[3,4], [1,4]], columns=['Column1','Column2']))

    @patch('utils.sql_utils.pymssql.connect')
    def test_no_return(self, mock_connect):
        """
        Tests a simple SQL query execution
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        query = "Select * from table"
        with self.assertWarns(Warning):
            results = su.run_sql_query(query=query)
            mock_cursor.execute.assert_called_once_with(query)
            mock_cursor.fetchall.assert_called_once()
            self.assertIsNone(results)

    @patch('utils.sql_utils.pymssql.connect')
    def test_exception(self, mock_connect):
        """
        Tests a for when a operationalerror is thrown
        """
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.__enter__.return_value = None
        mock_cursor.rowcount = -1
        mock_cursor.fetchall.side_effect = pymssql.OperationalError("executed statement has no resultset")
        mock_cursor.description = [['Column1'], ['Column2']]
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        query = "Select * from table"

        results = su.run_sql_query(query=query)
        mock_cursor.execute.assert_called_once_with(query)
        mock_cursor.fetchall.assert_called_once()
        self.assertIsNone(results)

class TestGenerateArgStrings(unittest.TestCase):
    """
    Test generate_arg_strings function
    """
    def test_basic_args(self):
        """
        Basic test using a string column and a int column
        """
        args = OrderedDict()
        args['c1'] = 'v1'
        args['c2'] = 1
        result = su.generate_arg_strings(args)
        self.assertEqual(result, "@c1='v1', @c2=1")

class TestGetClassMap(unittest.TestCase):
    """
    Test get_class_map function.
    """
    @patch('utils.sql_utils.run_sql_query')
    def test_get_map(self, mock_sql):
        """
        Tests retrieving a basic class map.
        """
        class_map = ['{"a":1,"b":2}'] # For example, integers from 1 to 10

        # Generate some other values
          # Random values

        # Create a DataFrame with the integers in the first column and other values in subsequent columns
        df = pd.DataFrame({'class_map': class_map})
        mock_sql.return_value = df
        model_id = 1
        result = su.get_class_map(model_id)
        self.assertIsInstance(result, dict)

    def test_value_error(self):
        """
        Tests passing a strin instead of int into function.
        """
        with self.assertRaises(ValueError):
            su.get_class_map('one')


class TestMapProbsColumn(unittest.TestCase):
    """
    Test map_probs_column function.
    """
    @patch('utils.sql_utils.get_class_map')
    def test_map_probs_column(self, mock_class_map):
        """
        Tests basic map probs functionality.
        """
        mock_class_map.return_value = {0: 'class_1', 1: "class_2"}
        # Sample dictionaries
        entry1 = {0: .5, 1: .5}
        entry2 = {0: .1, 1: .9}

        # Convert dictionaries to string representations
        entry1_str = str(entry1)
        entry2_str = str(entry2)

        # Create a pandas Series
        prob_columns = pd.Series([entry1_str, entry2_str])
        model_id = 1

        e_dict1 = {'class_1': .5, "class_2": .5}
        e_dict2 = {'class_1': .1, "class_2": .9}
        expected = pd.Series([e_dict1, e_dict2])
        result = su.map_probs_column(model_id, prob_columns)

        self.assertTrue(result.equals(expected))

    @patch('utils.sql_utils.get_class_map')
    def test_arg_value_error(self, mock_class_map):
        """
        Tests error throwing.
        """
        mock_class_map.return_value = {0: 'class_1', 1: "class_2"}
        # Sample dictionaries
        entry1 = {0: .5, 1: .5}
        entry2 = {0: .1, 1: .9}

        # Convert dictionaries to string representations
        entry1_str = entry1
        entry2_str = entry2

        # Create a pandas Series
        prob_columns = pd.Series([entry1_str, entry2_str])
        model_id = 1

        with self.assertRaises(ValueError):
            su.map_probs_column(model_id, 'prob_columns')
        with self.assertRaises(ValueError):
            su.map_probs_column(model_id, prob_columns)
        

class TestGenerateRandomEvaluationSet(unittest.TestCase):
    """
    Test generate_random_evaluation_set function.
    """
    @patch('utils.sql_utils.execute_stored_procedure')
    def test_basic_functionality(self, mock_stored_procedure):
        """
        Tests basic functionality for random evaluation set generation.
        """
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'Age': [25, 30, 35, 40, 45],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df

        su.generate_random_evaluation_set()
        mock_stored_procedure.assert_called_once()


    @patch('utils.sql_utils.execute_stored_procedure')
    def test_value_errors(self, mock_stored_procedure):
        """
        Tests basic functionality for random evaluation set generation.
        """
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'Age': [25, 30, 35, 40, 45],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        mock_stored_procedure.return_value = df
        with self.assertRaises(ValueError):
            su.generate_random_evaluation_set(train_ids=4)
        with self.assertRaises(ValueError):
            su.generate_random_evaluation_set(train_ids=[4,'5'])
        with self.assertRaises(ValueError):
            su.generate_random_evaluation_set(test_size=-100)

    @patch('utils.sql_utils.execute_stored_procedure')
    @patch('utils.sql_utils.warnings.warn')
    def test_warning_result(self, mock_warn, mock_stored_procedure):
        """
        Tests warnings for generating a random evaluation set.
        """
        def warning_side_effect(*args, **kwargs):
            # List of warning messages
            warnings_list = [
                "arguments returned empty",
                "Second warning message",
                "Third warning message"
            ]
            # Raise each warning in the list
            for warning_message in warnings_list:
                warnings.warn(warning_message, UserWarning)

        # Create DataFrame
        # mock_stored_procedure.return_value = df
        mock_stored_procedure.side_effect = warning_side_effect
        su.generate_random_evaluation_set()
        self.assertEqual(mock_warn.call_count, 3)

    @patch('utils.sql_utils.execute_stored_procedure')
    def test_return_none(self, mock_stored_procedure):
        """
        Tests  functionality for returning a none df.
        """
        mock_stored_procedure.return_value = None

        results = su.generate_random_evaluation_set()
            # Check if the warning was captured
        self.assertIsNone(results)

class TestUpdateScores(unittest.TestCase):
    """
    Tests for the update score function.
    """
    @patch('utils.sql_utils.run_sql_query')
    def test_basic_functionality(self, mock_sql_query):
        """
        Tests  functionality for returning a none df.
        """
        mock_sql_query.return_value = None
        self.assertIsNone(su.update_scores(i_ids=[1]))
        # self.assertIsNotNone(result)
        mock_sql_query.assert_called_once()

    @patch('utils.sql_utils.run_sql_query')
    @patch('utils.sql_utils.warnings.warn')
    def test_warning_result(self, mock_warn, mock_sql_query):
        """
        Tests warnings for generating a random evaluation set.
        """
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'Age': [25, 30, 35, 40, 45],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        mock_sql_query.return_value = df
        su.update_scores(i_ids=[1])
        self.assertEqual(mock_warn.call_count, 1)

    @patch('utils.sql_utils.run_sql_query')
    def test_errors(self, mock_sql_query):
        """
        Tests warnings for generating a random evaluation set.
        """
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'Age': [25, 30, 35, 40, 45],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        mock_sql_query.return_value = df

        with self.assertRaises(ValueError):
            su.update_scores(i_ids=[1],label_weight=3.4)
        with self.assertRaises(ValueError):
            su.update_scores(i_ids=[1,'23'])

class TestChunky(unittest.TestCase):
    """
    Tests for chunky.
    """
    def test_basic_function(self):
        """
        Tests the basic functionality of chunky.
        """
        result = su.chunky(lst=[1,2,3,4],n=1)
        self.assertIsInstance(result, Iterable)

        for i in result:
            self.assertIsInstance(i, list)

class TestDeleteLabelsCleanup(unittest.TestCase):
    """
    Tests for deleting labels cleanup
    """
    @patch('utils.sql_utils.run_sql_query')
    @patch('utils.sql_utils.update_scores')
    def test_basic_functionality(self, mock_update_scores, mock_sql_query):
        """
        Tests  functionality for returning a none df.
        """
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
            'W_COUNT': [25, 30, 35, 40, 45],
            'I_ID': [9,1,2,3,4]
        }
        data2 = {
            'Name': ['Alice', 'David', 'Emma'],
            'W_COUNT': [1,  2, 2],
            'I_ID': [9,3,4]
        }
        data3 = {
            'Name': ['Alice', 'David', 'Emma'],
            'W_COUNT': [1,  2, 2],
            'I_ID': [9,3,4]
        }
        # Create DataFrame
        df = pd.DataFrame(data)
        # Create DataFrame
        df2 = pd.DataFrame(data2)
        df3 = pd.DataFrame(data3)

        return_values = [df, df2, df3,None]

        mock_sql_query.side_effect = return_values
        self.assertIsNone(su.delete_labels_cleanup("fake query"))
        # self.assertIsNotNone(result)
        self.assertEqual(mock_sql_query.call_count, 3)
        self.assertEqual(mock_update_scores.call_count, 5)

    
if __name__ == '__main__':
    unittest.main()
