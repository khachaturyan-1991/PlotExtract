from functools import wraps

from flask import Flask

from controllers.__abstract__ import Controller


def methodroute(route, methods=['GET']):
    """
    Decorator to attach route information to methods.
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        
        controller = f.__qualname__.split('Controller.')[0].lower()
        wrapped.controller = '/api/' + controller if controller != 'index' else ''
        wrapped.route = route
        wrapped.methods = methods
        return wrapped
    return decorator


def routeapp(app: Flask, controller: Controller):
    """
    function to route with methodrout() decorated methods

    Args:
        app (Flask): instance of class Flask
        controller (Controller): instance of class Controller
    """
    for method in dir(controller):
        attr = getattr(controller, method)
        if callable(attr):
            if hasattr(attr, 'route'):
                route = f'{attr.controller}{attr.route}'
                methods = attr.methods
                # Register the route
                app.add_url_rule(route, view_func=attr, methods=methods)
