from flask import Flask
from controllers.plot_controller import PlotController
from controllers.index_controller import IndexController
from controllers.util import routeapp
from lib import injector
import logging

CONTROLLERS = [
    PlotController,
    IndexController
]

logger = logging.getLogger(__name__)


class Bootstrap:
    """
    A class for initializing and configuring components or systems.

    The `Bootstrap` class is typically used to set up and configure various aspects
    of an application or system, such as registering routes, setting up services,
    or initializing dependencies.

    This class provides a way to encapsulate the initialization logic, making it
    easier to manage and maintain the setup process.
    """
    def __init__(self):
        """
        initializes the interface
        """
        self._injector = injector.Injector()
        self.configure_dp_injector()

    
    def configure_dp_injector(self):
        """
        configures the dependency injector
        """
        pass
        # self.__injector.binder.bind(Class, to = self.__bacnet, scope=singleton)


    def get_controllers(self) -> list:
        """
        returns controller instances
        """
        return [self._injector.get(controller) for controller in CONTROLLERS]
    
    def run_routing(self, app: Flask) -> None:
        """
        routes the app
        """
        controllers = self.get_controllers()
        
        for controller in controllers:
            routeapp(app, controller)

    def run_webservice(self) -> None:
        """
        runs the webservice
        """
        app = Flask(__name__)

        self.run_routing(app)

        app.run(port=5001, debug=False)
