import unittest
import pandas as pd
import numpy as np
import logging
from io import StringIO

from datacleaner.datacleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cleaner = DataCleaner()
        
        # Define some example cleaner functions
        def remove_negatives(x):
            return np.maximum(0, x)
            
        def convert_to_int(x):
            return np.round(x).astype(int)
            
        def trim_outliers(x):
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return np.clip(x, lower_bound, upper_bound)
            
        # Register cleaners
        self.cleaner.add_cleaner(remove_negatives)
        self.cleaner.add_cleaner(convert_to_int)
        self.cleaner.add_cleaner(trim_outliers)
        
        # Create test data
        self.test_df = pd.DataFrame({
            'variable': ['heart_rate', 'sbp', 'heart_rate', 'sbp', 'glucose', 'glucose'],
            'value': [-5, 200, 80, 90, 120, 500],
            'time': pd.date_range(start='2023-01-01', periods=6, freq='H')
        })
        
    def test_add_cleaner(self):
        """Test adding a new cleaner function."""
        def new_cleaner(x):
            return x * 2
            
        self.cleaner.add_cleaner(new_cleaner)
        self.assertIn('new_cleaner', self.cleaner.cleaners)
        
        # Test that function works properly
        result = self.cleaner.apply_cleaner(np.array([1, 2, 3]), 'new_cleaner')
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))
        
    def test_remove_cleaner(self):
        """Test removing a cleaner function."""
        self.cleaner.remove_cleaner('remove_negatives')
        self.assertNotIn('remove_negatives', self.cleaner.cleaners)
        
        # Test that trying to use removed cleaner raises an error
        with self.assertRaises(ValueError):
            self.cleaner.apply_cleaner(np.array([1, 2, 3]), 'remove_negatives')
            
    def test_update_existing_cleaner(self):
        """Test updating a cleaner function."""
        def remove_negatives(x):  # Redefine the cleaner with the same name
            return np.where(x < 0, 0, x * 2)  # Changed behavior
            
        self.cleaner.update_cleaner(remove_negatives)
        result = self.cleaner.apply_cleaner(np.array([-1, 0, 1]), 'remove_negatives')
        np.testing.assert_array_equal(result, np.array([0, 0, 2]))

    
    def test_update_absent_cleaner(self):
        """Test updating a cleaner function."""
        def new_remove_negatives(x):  # Redefine the cleaner with the same name
            return np.where(x < 0, 0, x * 2)  # Changed behavior
            
        self.cleaner.update_cleaner(new_remove_negatives)
        result = self.cleaner.apply_cleaner(np.array([-1, 0, 1]), 'new_remove_negatives')
        np.testing.assert_array_equal(result, np.array([0, 0, 2]))

        
    def test_add_variable(self):
        """Test adding a variable with cleaners."""
        self.cleaner.add_variable('heart_rate', ['remove_negatives', 'convert_to_int'])
        self.assertIn('heart_rate', self.cleaner.variables)
        self.assertEqual(self.cleaner.variables['heart_rate'], ['remove_negatives', 'convert_to_int'])
        
        # Test adding with a callable
        def new_cleaner(x):
            return x * 2
            
        self.cleaner.add_variable('sbp', new_cleaner)
        self.assertIn('sbp', self.cleaner.variables)
        self.assertIn('new_cleaner', self.cleaner.cleaners)
        self.assertEqual(self.cleaner.variables['sbp'], ['new_cleaner'])
        
    def test_remove_variable(self):
        """Test removing a variable."""
        self.cleaner.add_variable('heart_rate', 'remove_negatives')
        self.cleaner.remove_variable('heart_rate')
        self.assertNotIn('heart_rate', self.cleaner.variables)
        
    def test_update_variable(self):
        """Test updating a variable's cleaners."""
        self.cleaner.add_variable('heart_rate', 'remove_negatives')
        
        # Test replacing cleaners
        self.cleaner.update_variable('heart_rate', 'convert_to_int')
        self.assertEqual(self.cleaner.variables['heart_rate'], ['convert_to_int'])
        
        # Test appending cleaners
        self.cleaner.update_variable('heart_rate', 'trim_outliers', append_cleaners=True)
        self.assertEqual(self.cleaner.variables['heart_rate'], ['convert_to_int', 'trim_outliers'])
        
    def test_call_with_column_variables(self):
        """Test cleaning data using column for variables."""
        # Register variables
        self.cleaner.add_variable('heart_rate', ['remove_negatives', 'convert_to_int'])
        self.cleaner.add_variable('sbp', 'trim_outliers')
        
        # Clean data
        result = self.cleaner(self.test_df, value_column='value', variable_column='variable')
        
        # Check heart_rate cleaning
        heart_rate_values = result[result['variable'] == 'heart_rate']['value'].tolist()
        self.assertEqual(heart_rate_values, [0, 80])
        
        # Check sbp cleaning
        sbp_values = result[result['variable'] == 'sbp']['value'].tolist()
        self.assertEqual(len(sbp_values), 2)  # Should still have two values

    def test_call_with_variables_as_columns(self):
        """Test cleaning a DataFrame where variables are column names directly."""
        # Create a DataFrame with columns that match registered variable names
        df = pd.DataFrame({
            'heart_rate': [-5, 80, 90],
            'sbp': [200, 90, 110],
            'time': pd.date_range(start='2023-01-01', periods=3, freq='H')
        })
        
        # Register variables
        self.cleaner.add_variable('heart_rate', ['remove_negatives', 'convert_to_int'])
        self.cleaner.add_variable('sbp', 'trim_outliers')
        
        # Clean data with variables as column names (both value_column and variable_column are None)
        result = self.cleaner(df, value_column=None, variable_column=None)
        
        # Check heart_rate cleaning (remove negatives and convert to int)
        self.assertEqual(result['heart_rate'].tolist(), [0, 80, 90])
        
        # Check sbp cleaning (trim outliers)
        self.assertEqual(len(result['sbp']), 3)
        # Verify sbp values are within expected range
        q1, q3 = np.percentile([90, 110, 200], [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        self.assertTrue(all(val <= upper_bound for val in result['sbp']))
        
        # Check time column is untouched (not a registered variable)
        pd.testing.assert_series_equal(df['time'], result['time'])

        
    def test_inplace_cleaning(self):
        """Test inplace cleaning."""
        # Register variables
        self.cleaner.add_variable('heart_rate', 'remove_negatives')
        
        # Clean data inplace
        df_copy = self.test_df.copy()
        result = self.cleaner(df_copy, value_column='value', variable_column='variable', inplace=True)
        
        # Check that result is the same object as df_copy
        self.assertIs(result, df_copy)
        
        # Check that heart_rate cleaning worked
        self.assertEqual(df_copy[df_copy['variable'] == 'heart_rate']['value'].tolist()[0], 0)
        
    def test_no_registered_variables(self):
        """Test behavior when no registered variables are found."""
        # Capture log messages
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logging.getLogger().addHandler(handler)
        
        # Clean data with no registered variables
        result = self.cleaner(self.test_df, value_column='value', variable_column='variable')
        
        # Check log message
        self.assertIn("No registered variables found", log_capture.getvalue())
        
        # Check that data is unchanged
        pd.testing.assert_frame_equal(result, self.test_df)
        
        # Clean up
        logging.getLogger().removeHandler(handler)
        
    def test_missing_value_column(self):
        """Test error when value column is missing."""
        with self.assertRaises(ValueError):
            self.cleaner(self.test_df, value_column='non_existent')
            
    def test_preserve_input_type(self):
        """Test that input types are preserved."""
        # Test with list
        test_list = [-5, 10, 15]
        result = self.cleaner.apply_cleaner(test_list, 'remove_negatives')
        self.assertIsInstance(result, list)
        self.assertEqual(result, [0, 10, 15])
        
        # Test with Series
        test_series = pd.Series([-5, 10, 15], name='test')
        result = self.cleaner.apply_cleaner(test_series, 'remove_negatives')
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, 'test')
        self.assertEqual(result.tolist(), [0, 10, 15])
        
    def test_error_on_invalid_input(self):
        """Test error on invalid input types."""
        # Test scalar input (not array-like)
        with self.assertRaises(TypeError):
            self.cleaner.apply_cleaner(5, 'remove_negatives')
            
        # Test 2D array input
        with self.assertRaises(ValueError):
            self.cleaner.apply_cleaner(np.array([[1, 2], [3, 4]]), 'remove_negatives')
            
    def test_str_representation(self):
        """Test string representation of DataCleaner."""
        str_repr = str(self.cleaner)
        self.assertIn("DataCleaner with", str_repr)
        self.assertIn("registered cleaner methods", str_repr)
        self.assertIn("remove_negatives", str_repr)
        self.assertIn("convert_to_int", str_repr)
        self.assertIn("trim_outliers", str_repr)


if __name__ == '__main__':
    unittest.main()