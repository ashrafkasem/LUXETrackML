
from abc import ABC, abstractmethod
class DataConnector(ABC):
    
    @abstractmethod
    def readfile(self, File):
        pass

    @abstractmethod
    def make_partial_df(self):
        pass

    @abstractmethod
    def make_truth(self):
        pass

    @abstractmethod
    def filter_data(self):
        pass