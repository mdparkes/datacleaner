# DataCleaner

A flexible utility for transforming data with custom functions.

## Features
* Register custom cleaner functions
* Associate specific cleaner functions with variables
* Sequentially apply multiple cleaners to a variable
* Works with `pd.DataFrame` objects with variables as column names or variable names stored in a single column (e.g. 
  multivariable time series data)
* Preserves original data type of the input

## Details

The `DataCleaner` class facilitates transformation of data with user-defined functions. These "cleaner" functions should receive a 1D `ArrayLike` input and return the same. Instances of the DataCleaner class are callable on a `pd.DataFrame` object and apply pipelines of transformations to specific variables in the `DataFrame`.

The `DataCleaner` registers the names of variables to transform and a sequence of user-defined cleaner functions to apply to the variable. Variables and cleaner methods can be added, removed, and modified using the methods of `DataCleaner`.

`DataCleaner` also provides an `apply_cleaner` method that can be used to apply any registered cleaner function to an arbitrary 1D `ArrayLike` input.

## Installation

```shell
git clone https://github.com/mdparks/datacleaner/
cd /path/to/datacleaner
pip install .
```

## Usage
```python
import pandas as pd
import numpy as np
from datacleaner import DataCleaner

# Initialize a new DataCleaner instance
dc = DataCleaner()

# Create some test data (timeseries)
df = pd.DataFrame({
    'variable': ['heart_rate', 'temperature', 'heart_rate', 'temperature', 'glucose', 'glucose', 'temperature'],
    'value': [-5, 98.1, 80, 99.8, 120, 500, -98.6],
    'time': pd.date_range(start='2023-01-01', periods=7, freq='H')
})

# Define some custom cleaner functions
def remove_negatives(x):
    return np.maximum(0, x)

def f_to_c(x):
    return (x - 32) * 5 / 9

# Register the cleaner functions with the DataCleaner
dc.add_cleaner(remove_negatives)

# Register some variables to transform with the DataCleaner
dc.add_variable('heart_rate', 'remove_negatives')
dc.add_variable('temperature', ['remove_negatives', f_to_c])  # Automatically registers f_to_c

# View information about the DataCleaner
str(dc)

# Apply transformations (returns a copy by default, in-place modification is optional)
dc(df, variable_column='variable', value_column='value', inplace=False)

# Update a registered cleaner method
def remove_negatives(x)  # Must have the same name as the original
    return np.where(x < 0, 0, x)

dc.update_cleaner(remove_negatives)

# Remove a registered cleaner or variable
dc.remove_cleaner('f_to_c')
dc.update_variable('temperature', 'remove_negatives', append_cleaners=False)
# Add f_to_c back
dc.update_variable('temperature', f_to_c, append_cleaner=True)
```


