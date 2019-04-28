from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def eval(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def hessian(self, x):
        pass
