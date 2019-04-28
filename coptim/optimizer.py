from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, **kwargs):
        pass

    @abstractmethod
    def stopping_criteria(self, **kwargs):
        pass
