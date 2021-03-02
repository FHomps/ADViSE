from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, inp, label):
        pass
    
    @abstractmethod
    def evaluate(self, inp, label):
        pass
    
    @abstractmethod
    def saveToFile(self, filename):
        pass
    
    @abstractmethod
    def loadFromFile(self, filename):
        pass
    
    @abstractmethod
    def initWeights(self):
        pass
    
    @abstractmethod
    def __call__(self, inp):
        pass