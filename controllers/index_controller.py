 # Python System Imports
import os

# controller util imports
from controllers.util import methodroute
from controllers.__abstract__ import Controller
from flask import send_from_directory

# dependency injector import
from lib.injector import singleton

PUBLIC_PATH = "public/app/browser"
@singleton
class IndexController(Controller):
    """
    Controller for delivering ui files
    """
    def __init__(self) -> None:
        """
        Initialize the interface
        """
        pass
    

    @methodroute('/')
    def serve_homepage(self):
        """
        serves the ui index file
        """
        public_path = f'{os.path.dirname(__file__)}/../{PUBLIC_PATH}' if os.path.dirname(__file__) else f'./{PUBLIC_PATH}'

        return send_from_directory(public_path, 'index.html')

    @methodroute('/<path:filename>')
    def serve_static(self, filename):
        """
        returns any files according to their path
        """
        public_path = f'{os.path.dirname(__file__)}/../{PUBLIC_PATH}' if os.path.dirname(__file__) else f'./{PUBLIC_PATH}'
        
        if '.' in filename:
            return send_from_directory(public_path, filename)
    
        return send_from_directory(public_path, 'index.html')