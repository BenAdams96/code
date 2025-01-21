# enums.py
from enum import Enum

class Descriptor(Enum):
    WHIM = 'WHIM'
    GETAWAY = 'GETAWAY'

    def __str__(self):
        return self.value

class DatasetProtein(Enum):
    JAK1 = 'JAK1'
    GSK3 = 'GSK3'
    pparD = 'pparD'

    def __str__(self):
        return self.value