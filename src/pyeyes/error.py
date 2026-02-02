import sys
from functools import wraps

import panel as pn

pn.extension(notifications=True)
_ORIGINAL_EXCEPTHOOK = sys.excepthook


def error_handler_decorator(disp_duration_ms=3000):
    """Decorator: on exception, show Panel notification then re-raise."""

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
    """Show Panel warning notification."""
    pn.state.notifications.warning(msg, duration=duration_ms)


def global_error_handler(exc_type, exc_value, exc_traceback):
    """Excepthook: show Panel error and re-raise; KeyboardInterrupt passes through."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to exit gracefully
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    pn.state.notifications.error(f"An error occurred: {exc_value}", duration=0)
    raise exc_value


def install_pyeyes_error_handler():
    """Install global excepthook that shows Panel error notification."""
    sys.excepthook = global_error_handler


def uninstall_pyeyes_error_handler():
    """Restore default sys.excepthook."""
    sys.excepthook = _ORIGINAL_EXCEPTHOOK
