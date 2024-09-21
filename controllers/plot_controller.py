 # Python System Imports
import logging
import os

# controller util imports
from controllers.__abstract__ import Controller
from controllers.util import methodroute
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# dependency injector import
from lib.injector import inject, singleton
from services.plot_service import PlotService


logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'temp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@singleton
class PlotController(Controller):
    """
    Controller for delivering ui files
    """
    @inject
    def __init__(self, plot_service: PlotService) -> None:
        """
        Initialize the interface
        """
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        self._plot_service = plot_service
    
    def _allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @methodroute('/extract', methods=['POST'])
    def extract(self):
        """
        Endpoint for uploading an image file
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and self._allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            result = self._plot_service.extract_plot(file_path)
            return jsonify({
                'message': 'SUCCESS', 
                'result': True, 
                'detail': result 
            }), 201
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    @methodroute('/test')
    def test(self):
        """
        Endpoint for testing
        """
        return jsonify({'error': False, 'message': 'File type not allowed'})
