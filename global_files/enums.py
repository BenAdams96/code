# enums.py
from enum import Enum

class Model_classic(Enum):
    RF = 'RF'
    XGB = 'XGB'
    SVM = 'SVM'

    def __str__(self):
        return self.value

class Model_deep(Enum):
    DNN = 'DNN'

class Descriptor(Enum):
    WHIM = 'WHIM' , 114
    GETAWAY = 'GETAWAY' , 273

    def __new__(cls, value, descriptor_length):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._descriptor_length = descriptor_length       
        return obj

    @property
    def descriptor_length(self) -> int:
        return self._descriptor_length

    def __str__(self) -> str:
        return self.value

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