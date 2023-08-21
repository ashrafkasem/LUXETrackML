
from abc import ABC, abstractmethod
class DataConnector(ABC):
    
    @abstractmethod
    def readfile(self, File):
        pass

    @abstractmethod
    def make_hits(self):
        pass

    @abstractmethod
    def make_particles(self):
        pass

    @abstractmethod
    def make_truth(self):
        pass

    @abstractmethod
    def filter_data(self):
        pass