from abc import ABC, abstractmethod


class PlotProcessor(ABC):
    """
    Abstract class for plot processing classes
    """
    @abstractmethod
    def handle(self):
        pass
