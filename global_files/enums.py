# enums.py
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from typing import Union

class Model_classic(Enum):
    RF = ('RF',     RandomForestRegressor,          {'n_estimators': [100],
                                                    'max_depth': [5,8],
                                                    'min_samples_split': [2,5],
                                                    'min_samples_leaf': [2,5],
                                                    'max_features': ['sqrt']
                                                    })
    XGB = ('XGB',   XGBRegressor,                   {"n_estimators": [100, 200], 
                                                     "max_depth": [3, 10]})
    SVM = ('SVM',   SVR,                            {"C": [1, 10],
                                                     "kernel": ["linear", "rbf"]})
    
    model: Union[RandomForestRegressor, XGBRegressor, SVR]
    hyperparameter_grid: dict

    def __new__(cls, value, _model, _hyperparameter_grid):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._model = _model  # Store the model class (not an instance)
        obj.hyperparameter_grid = _hyperparameter_grid
        return obj

    @property
    def model(self):
        """Instantiate the model class with optional random_state."""
        # Check if random_state is applicable for the model type
        if self._model in [RandomForestRegressor, XGBRegressor]:
            print('random state')
            return self._model(random_state=42)  # Set random_state for models that support it
        return self._model()  # For SVM and other models that don't have random_state

    def __str__(self) -> str:
        return self.value  # Ensure `str()` outputs the `value`


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

class Model_deep(Enum):
    DNN = 'DNN'

class Descriptor(Enum):
    WHIM = 'WHIM' , 114
    GETAWAY = 'GETAWA' , 273

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
    JAK1 = 'JAK1', 615
    GSK3 = 'GSK3', 856
    pparD = 'pparD', 1125

    def __new__(cls, value, dataset_length):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._dataset_length = dataset_length        
        return obj

    @property
    def dataset_length(self) -> int:
        return self._dataset_length

    def __str__(self) -> str:
        return self.value