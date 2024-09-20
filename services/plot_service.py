from lib.injector import singleton

@singleton
class PlotService:
    """
    Service for plot bussiness logic
    """
    
    def __init__(self) -> None:
        """
        initializes the interface
        """
        pass

    def extract_plot(self, filepath) -> list:
        """
        extracts plots from an image file
        """
        pass