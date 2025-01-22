import sys
from functools import wraps

import panel as pn

pn.extension(notifications=True)


# Decorator to handle errors
def error_handler_decorator(disp_duration_ms=3000):
    def decorator(func):
        # This ensures the wrapper retains the metadata (including name) of the original function
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                pn.state.notifications.error(
                    f"An error occurred: {e}", duration=disp_duration_ms
                )
                # We should still print full traceback to console - otherwise it's hard to debug
                raise e

        return wrapper

    return decorator


def warning(msg, duration_ms=3000):
    pn.state.notifications.warning(msg, duration=duration_ms)


# This will catch any unhandled exceptions
def global_error_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to exit gracefully
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    pn.state.notifications.error(f"An error occurred: {exc_value}", duration=0)
    raise exc_value


sys.excepthook = global_error_handler
