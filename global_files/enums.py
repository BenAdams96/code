# enums.py
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from typing import Union
from itertools import product

class Model_classic(Enum):
    RF = 'RF', RandomForestRegressor, {'n_estimators': [100, 200], 'max_depth': [5, 10, 15], 
                     'min_samples_split': [5, 10, 15], 'min_samples_leaf': [2, 5, 10], 
                     'max_features': ['sqrt', 'log2']}

    XGB = 'XGB', XGBRegressor, {
                                'n_estimators': [100, 200],             # Number of boosting rounds
                                'max_depth': [3, 5, 10],            # Tree depth (controls complexity)
                                'learning_rate': [0.01, 0.05, 0.1],     # Learning rate (smaller = more robust)
                                'subsample': [0.7, 0.9],                # % of rows used per boosting round
                                'colsample_bytree': [0.6, 0.8],         # % of columns used per tree
                                'gamma': [0.8],                         # Minimum loss reduction to split
                                'alpha': [0, 0.1],                      # L1 regularization (sparsity)
                                'lambda': [1, 1.5],                     # L2 regularization (weight shrinkage)
                            }
    # XGB = 'XGB', XGBRegressor, {
    #                         'n_estimators': [200],             # Number of boosting rounds
    #                         'max_depth': [10],            # Tree depth (controls complexity)
    #                         'learning_rate': [0.1],     # Learning rate (smaller = more robust)
    #                         'subsample': [0.9],                # % of rows used per boosting round
    #                         'colsample_bytree': [0.8],         # % of columns used per tree
    #                         'gamma': [0.8],                         # Minimum loss reduction to split
    #                         'alpha': [0.1],                      # L1 regularization (sparsity)
    #                         'lambda': [1.5],                     # L2 regularization (weight shrinkage)
    #                         }
    SVM = 'SVM', SVR, {
                        'C': [0.1, 1,10],             # Controls the trade-off between smoothness and accuracy
                        'kernel': ['linear','rbf','poly'],
                        'degree': [2,3],                  # Only relevant for 'poly'; keeps it conservative
                        'gamma': ['scale', 'auto'],       # Controls the influence of a single training example (relevant for 'rbf' and 'poly')
                        'epsilon': [0.1, 0.2],            # Epsilon in the epsilon-SVR model; margin of tolerance
                        }

    model: Union[RandomForestRegressor, XGBRegressor, SVR]
    hyperparameter_grid: dict

    def __new__(cls, value, _model, _hyperparameter_grid):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._model = _model  # Store the model class
        obj._hyperparameter_grid = _hyperparameter_grid  # Store both grids
        return obj

    @property
    def model(self):
        """Instantiate the model class with optional random_state."""
        if self._model in [RandomForestRegressor, XGBRegressor]:
            return self._model(random_state=42)  # Set random_state for reproducibility
        return self._model()
    
    @property
    def hyperparameter_grid(self):
        """Return the hyperparameter grid for the current model."""
        return self._hyperparameter_grid
    
    def __str__(self) -> str:
        return self.value  # Ensure `str()` outputs the model name

class Model_deep(Enum):
    DNN = 'DNN' ,   {"lr": [0.001],
                    "hidden_layers": [[128, 64, 32],[64, 32, 16],[128, 64],[256, 128, 64]],
                    "dropout": [0.3],#, 0.2, 0.4],
                    } #20 in total
                    #{"lr": [0.001],
                    #"hidden_layers": [[128, 64, 32]],
                    #"dropout": [0.2],
                    #} #20 in total
                    
    
    LSTM = 'LSTM' , {"hidden_size": [32,64],  # Number of LSTM units per layer
                    "num_layers": [1,2,3],  # Number of stacked LSTM layers
                    "dropout": [0.1],  # Dropout to prevent overfitting
                    "learning_rate": [1e-4],  # Learning rate for Adam/SGD
                    "weight_decay": [1e-5],  # L2 regularization
                    } #108 in total
    
    hyperparameter_grid: dict

    def __new__(cls, value, _hyperparameter_grid):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.hyperparameter_grid = _hyperparameter_grid
        return obj

    def __str__(self) -> str:
        return self.value

    @property
    def get_hyperparameter_grid(self):
        """Filters the hyperparameter grid based on model-specific rules."""

        grid = self.hyperparameter_grid
        if not grid:
            return []

        all_combinations = [
            dict(zip(grid.keys(), values))
            for values in product(*grid.values())
        ]

        # Apply LSTM-specific filtering
        if self == Model_deep.LSTM:
            return [
                hp for hp in all_combinations if not (hp["num_layers"] == 1 and hp["dropout"] != 0.0)
            ]
        return all_combinations  # No filtering for DNN


# class Model_classic(Enum):
#     RF = ('RFaa', RandomForestRegressor, {"n_estimators": [100, 200], "max_depth": [10, 20]})
#     XGB = ('XGBaa', XGBRegressor, {"n_estimators": [100, 200], "max_depth": [3, 10]})
#     SVM = ('SVRaa', SVR, {"C": [1, 10], "kernel": ["linear", "rbf"]})

#     def __new__(cls, name, model_class, hyperparameter_grid):
#         # Assign all values to the object during creation
#         obj = object.__new__(cls)
#         obj._value_ = name
#         obj._model_class = model_class
#         obj._hyperparameter_grid = hyperparameter_grid
#         return obj

#     @property
#     def model_class(self):
#         """ Return the model class (so you can instantiate it when needed). """
#         return self._model_class

#     @property
#     def hyperparameter_grid(self):
#         """ Return the hyperparameter grid. """
#         return self._hyperparameter_grid

#     def __str__(self) -> str:
#         return self.value

class Descriptor(Enum):
    WHIM = 'WHIM' , 114
    GETAWAY = 'GETAWAY' , 273

    descriptor_length: int
    
    def __new__(cls, value, _descriptor_length):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.descriptor_length = _descriptor_length
        return obj
    
    def __str__(self) -> str: #is for pv.DESCRIPTOR becomes the string 'WHIM'
        return self.value  # Ensure `str()` outputs the `value`


# class Circle:
#     def __init__(self, radius):
#         self._radius = radius  # Private attribute
    
#     @property
#     def area(self):
#         # Calculate area dynamically
#         return 3.14159 * (self._radius ** 2)
# class Descriptor(Enum):
#     WHIM = 'WH', 114
#     GETAWAY = 'GETAWA', 273

#     @property
#     def descriptor_length(self) -> int:
#         # Access the second value in the tuple, which is the integer value
#         return self.value[1]

#     def __str__(self) -> str:
#         return self.value[0]

class DatasetProtein(Enum):
    JAK1 = 'JAK1', 615, 5
    GSK3 = 'GSK3', 856, 5
    pparD = 'pparD', 1125, 5
    CLK4 = 'CLK4', 731, 5




    def __new__(cls, value, dataset_length, num_of_splits):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._dataset_length = dataset_length
        obj._num_of_splits = num_of_splits      
        return obj

    @property
    def dataset_length(self) -> int:
        return self._dataset_length
    
    @property
    def num_of_splits(self) -> int:
        return self._num_of_splits

    def __str__(self) -> str:
        return self.value