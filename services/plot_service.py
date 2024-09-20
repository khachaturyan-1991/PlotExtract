import numpy as np
import cv2
from lib.injector import singleton, inject
from plot_gear import scanner

@singleton
class PlotService:
    """
    Service for plot bussiness logic
    """
    @inject
    def __init__(self, plot_scanner: scanner.PlotScanner) -> None:
        """
        initializes the interface
        """
        self._plot_scanner = plot_scanner

    def extract_plot(self, filepath: str) -> dict:
        """
        extracts plots from an image file

        Args:
            filepath (str): path of the plot image

        Returns:

        """
        img = cv2.imread(filepath).astype(np.float32)
        img = cv2.resize(img, (296, 296), interpolation=cv2.INTER_CUBIC)
        return self._plot_scanner.handle(img)
