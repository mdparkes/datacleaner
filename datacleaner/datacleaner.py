import functools
import logging
import pandas as pd
import numpy as np
import warnings

from typing import List, Optional, Union
from numpy.typing import ArrayLike


class DataCleaner:

    def __init__(self):

        super().__init__()
        self.cleaners = set()  # A set of registered cleaner functions
        self.variables = dict()  # Maps from variable name to cleaner name

    @staticmethod
    def _preserve_input_type(method):
        """Attempts to preserve the input type of data passed to the cleaner method."""

        @functools.wraps(method)
        def wrapper(self, x: ArrayLike, cleaner_name: str) -> ArrayLike:
            
            # Store original type and other metadata
            original_type = type(x)
            is_series = isinstance(x, pd.Series)
            series_name = x.name if is_series else None
            series_index = x.index if is_series else None
            
            # Apply the cleaner method
            result = method(self, x, cleaner_name)
            
            # Try to restore original type
            if is_series:
                return pd.Series(result, index=series_index, name=series_name)
            
            elif original_type != np.ndarray:
                try:
                    return original_type(result)
                except ValueError:
                    return result  # If conversion fails, return the result as-is
                
            return result
        
        return wrapper
    
    @staticmethod
    def _update_cleaner_registry(method):

        @functools.wraps(method)
        def wrapper(self, cleaner: Union[callable, str]) -> None:

            if method.__name__ == 'add_cleaner':
                self._assert_cleaner_registerable(cleaner)
                method(self, cleaner)
                self.cleaners.add(cleaner.__name__)
            
            elif method.__name__ == 'remove_cleaner':
                if not isinstance(cleaner, str):
                    raise TypeError(f"'{cleaner}' must be a string, got type {type(cleaner)}")
                cleaner_name = cleaner
                if cleaner_name not in self.cleaners:
                    warnings.warn(f"No registered method called '{cleaner_name}'", UserWarning)
                else:
                    method(self, cleaner)
                    self.cleaners.remove(cleaner_name)

        return wrapper
    
    def _assert_cleaner_registered(self, cleaner: str) -> None:
        """Raise an exception if the cleaner is not registered with the DataCleaner instance."""

        if isinstance(cleaner, str):
            cleaner_name = cleaner
            if cleaner_name not in self.cleaners:
                raise ValueError(f"Cleaner '{cleaner_name}' is not registered. "
                                 "To register a new cleaner, use the 'DataCleaner.add_cleaner' method "
                                 f"or pass {cleaner_name} as a callable.")
            
        else:
            raise TypeError(f"{cleaner} must be a string, got type {type(cleaner)}")

    def _assert_cleaner_registerable(self, cleaner: callable) -> None:
        """Raise an exception if the cleaner cannot be registered with the DataCleaner instance."""

        if callable(cleaner):
            cleaner_name = cleaner.__name__
            if cleaner_name in self.cleaners:
                raise ValueError(f"'{cleaner_name}' is already registered. "
                                 f"To overwrite the registered definition of {cleaner_name}, "
                                 "use the 'DataCleaner.update_cleaner' method.")
            
        else:
            raise TypeError(f"'cleaner' must be a callable function, got type {type(cleaner)}")

    def _assert_variable_registered(self, variable_name: str) -> None:
        """Raise an exception if the variable is not registered with the DataCleaner instance."""

        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' is not registered. "
                             "To register a new variable, use the 'DataCleaner.add_variable' method.")

    def _assert_variable_registerable(self, variable_name: str) -> None:
        """Raise an exception if the variable cannot be registered with the DataCleaner instance."""

        if variable_name in self.variables:
            raise ValueError(f"Variable '{variable_name}' is already registered. "
                             "To update its registered cleaners, use the 'DataCleaner.update_variable' method.")

    def _register_cleaner_to_variable(self, variable_name: str, cleaner: Union[str, callable]) -> None:

        self._assert_variable_registered(variable_name)
        if callable(cleaner):
            cleaner_name = cleaner.__name__
            self.add_cleaner(cleaner)  # Register cleaner, raise an exception if the cleaner cannot be registered

        elif isinstance(cleaner, str):
            cleaner_name = cleaner
            self._assert_cleaner_registered(cleaner_name)  # Raises an exception if the cleaner is not registered

        else:
            raise TypeError(f"{cleaner} must be a string or callable, got type {type(cleaner)}")

        self.variables[variable_name].append(cleaner_name)

    @staticmethod
    def _validate_array(x: ArrayLike) -> ArrayLike:
        """Raise an exception if x is not a 1-D ArrayLike object."""

        # Check whether x is array-like
        if not hasattr(x, '__len__') and not hasattr(x, '__array__'):
            raise TypeError(f"Input must be array-like, got {type(x)}")

        x_ndim = np.atleast_1d(np.asarray(x)).ndim
        if x_ndim != 1:
            raise ValueError(f"Input must be 1-D, got {x_ndim}-D")
        
        return x

    @_update_cleaner_registry
    def add_cleaner(self, cleaner: callable) -> None:
        """
        Add a new cleaner function to the DataCleaner instance.
        Args:
            cleaner (callable): The cleaner function to add as a method of the DataCleaner instance.
        """

        cleaner_name = cleaner.__name__
        setattr(self, cleaner_name, cleaner)

    @_update_cleaner_registry
    def remove_cleaner(self, cleaner_name: str) -> None:
        """
        Remove a cleaner function from the DataCleaner instance.

        Args:
            cleaner_name (str): The name of the cleaner function to remove.
        """

        delattr(self, cleaner_name)

    @_update_cleaner_registry
    def update_cleaner(self, cleaner: callable) -> None:
        """
        Update an existing cleaner function in the DataCleaner instance. Behaves like 'add_cleaner' if the cleaner is
        not already registered.

        Args:
            cleaner (callable): The cleaner function to update. Must have the same name as an existing cleaner.
        """

        cleaner_name = cleaner.__name__
        if cleaner_name in self.cleaners:
            self.remove_cleaner(cleaner_name)
        self.add_cleaner(cleaner)

    def add_variable(self, variable_name: str, cleaners: Union[str, callable, List[Union[str, callable]]]) -> None:
        """Add a new variable to the DataCleaner instance and specify the cleaner function to use for it.
        
        Args:
            variable_name (str): The name of the variable to add.
            cleaners (List[str or callable]): The name(s) of a registered cleaner function or callable cleaner 
                function(s) to register with the DataCleaner and use for this variable. If a list is provided, the 
                DataCleaner will apply the cleaners in the listed order when called on data.
        
        Raises:
            ValueError: If the variable is already registered.
            ValueError: If a cleaner function is not registered with the DataCleaner instance and a string is provided,
                or if a callable is provided and it cannot be registered to the DataCleaner instance.
            TypeError: If the arguments have unexpected types.
        """
        self._assert_variable_registerable(variable_name)  # Raises an exception if the variable already registered
        self.variables[variable_name] = []

        if isinstance(cleaners, str) or callable(cleaners):
            cleaners = [cleaners]

        for cleaner in cleaners:
            self._register_cleaner_to_variable(variable_name, cleaner)
    
    def remove_variable(self, variable_name: str) -> None:
        """
        Remove a variable from the DataCleaner instance.

        Args:
            variable_name (str): The name of the variable to remove.
        
        Raises:
            ValueError: If the variable is not registered
        """

        self._assert_variable_registered(variable_name)
        del self.variables[variable_name]  # Remove the variable from the DataCleaner instance

    def update_variable(self, variable_name: str, cleaners: Union[str, callable, List[Union[str, callable]]], 
                        append_cleaners: bool = False) -> None:
        """
        Update an existing variable in the DataCleaner instance.

        Args:
            variable_name (str): The name of the variable to update.
            cleaners (List[str or callable]): The name(s) of a registered cleaner function or callable cleaner 
                function(s) to register with the DataCleaner and use for this variable. If a list is provided, the 
                DataCleaner will apply the cleaners in the listed order when called on data. 
            append_cleaners (bool): If True, append the new cleaner(s) to the existing list of cleaners for this
                variable. If False (default), replace the existing cleaner(s) with the new one(s).
        
        Raises:
            ValueError: If the variable is not registered
            ValueError: If a cleaner function is not registered with the DataCleaner instance and a string is provided,
                or if a callable is provided and it cannot be registered to the DataCleaner instance.
            TypeError: If the arguments have unexpected types.
        """

        self._assert_variable_registered(variable_name)  # Raises an exception if the variable is not registered

        if isinstance(cleaners, str) or callable(cleaners):
            cleaners = [cleaners]

        if append_cleaners:
            for cleaner in cleaners:
                self._register_cleaner_to_variable(variable_name, cleaner)
        else:
            # Remove all existing cleaners for this variable
            self.variables[variable_name] = []
            for cleaner in cleaners:
                self._register_cleaner_to_variable(variable_name, cleaner)

    @_preserve_input_type
    def apply_cleaner(self, x: ArrayLike, cleaner_name: str) -> ArrayLike:
        """
        Apply a registered cleaner function to the input data.

        Args:
            x: A 1-D ArrayLike object (e.g. flat list, 1-D np.ndarray, or pd.Series) containing the data to be cleaned.
            cleaner_name (str): The name of the registered cleaner function to apply to x.

        Returns:
            ArrayLike: The transformed data.
        
        Raises:
            ValueError: If the cleaner is not registered with the DataCleaner instance.
            ValueError: If the input to cleaner is not a 1-D ArrayLike object, or if the cleaner does not return a 1-D
                ArrayLike object.
            TypeError: If the input data is not an ArrayLike object.
        """

        self._assert_cleaner_registered(cleaner_name)
        cleaner_func = getattr(self, cleaner_name)
        # Validate the input data
        x = self._validate_array(x)
        # Apply the cleaner function
        x = cleaner_func(x)
        # Validate the output data
        x = self._validate_array(x)

        return x
    
    def __str__(self):

        n_cleaners = len(self.cleaners)
        n_variables = len(self.variables)
        string = f"DataCleaner with {n_cleaners} registered cleaner methods and {n_variables} variables.\n"
        if self.cleaners:
            string += "Registered cleaners:\n"
            for cleaner in sorted(list(self.cleaners)):
                string += f"    - {cleaner}\n"

        return string

    def __repr__(self):

        return f"DataCleaner(cleaners={self.cleaners}, variables={self.variables})"
    
    def __call__(self, x: pd.DataFrame, value_column: Optional[str], 
                 variable_column: Optional[str] = None, inplace: bool = False) -> pd.DataFrame:
        """
        Clean the input data by applying the cleaner functions in the order they were registered to
        variables.

        Args:
            x (pd.DataFrame): The input data to be cleaned.
            value_column (str, optional): The name of the column containing the values to be cleaned. Must be supplied
                if `variable_column` is supplied.
            variable_column (str, optional): The name of a column containing the variable names. If `None` (default), 
                `DataCleaner` assumes that the registered variables are column names. Must be supplied if 
                `value_column` is supplied.
            inplace (bool): If `True`, modify the input `DataFrame` in place. If `False` (default), return a new 
                `DataFrame`.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            ValueError: If `variable_column` is not found in the input `DataFrame`.
            ValueError: If `value_column` is not found in the input `DataFrame`.
            ValueError: If `variable_column` is not supplied when `value_column` is supplied.
        """

        # Check arguments
        if variable_column is not None:
            if variable_column not in x.columns:
                raise ValueError(f"Variable column '{variable_column}' not found in the input DataFrame.")
            if value_column is None:
                raise ValueError("Value column must be supplied if variable column is supplied.")
        
        if value_column is not None:
            if value_column not in x.columns:
                raise ValueError(f"Value column '{value_column}' not found in the input DataFrame.")
            if variable_column is None:
                raise ValueError("Variable column must be supplied if value column is supplied.")
        
        if not inplace:
            # Create a deep copy of the DataFrame to avoid modifying the original
            x = x.copy(deep=True)  
        
        # Get the names of registered variables in the DataFrame
        x_variables = x[variable_column].unique() if variable_column is not None else x.columns
        registered_variables = [var for var in x_variables if var in self.variables]

        # If no registered variables are found, return the input DataFrame as-is
        if not registered_variables:
            logging.warning(f"No registered variables found in the input DataFrame, returning as-is")
            return x
        
        for variable in registered_variables:

            if variable_column is None:

                for cleaner in self.variables[variable]:
                    input_data = x[variable].copy()
                    cleaned_data = self.apply_cleaner(input_data, cleaner)
                    x[variable] = cleaned_data
                
            else:

                sel = x[variable_column] == variable
                for cleaner in self.variables[variable]:
                    input_data = x.loc[sel, value_column].copy()
                    cleaned_data = self.apply_cleaner(input_data, cleaner)
                    x.loc[sel, value_column] = cleaned_data

        return x
