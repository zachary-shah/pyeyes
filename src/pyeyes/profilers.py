import cProfile
import os
import warnings
from functools import wraps
from time import perf_counter


def profile_decorator(
    enable: bool = True,
    verbose: bool = True,
    save: bool = False,
    save_path: str = "./profs/function_profile",
    max_profs_size: int = 25,
) -> callable:
    """
    Decorator for profiling functions

    Parameters
    ----------
    verbose : bool, optional
        Whether to print basic profile of timing of function call, by default True
    save : bool, optional
        Whether to save the profiling information using cProfile, by default False
    save_path : str, optional
        Path to save the profiling information, by default "./profs/view_profile"
    max_profs_size : int, optional
        Maximum number of profiles to save, by default 3
    """

    def decorator(func):

        if not enable:
            return func

        if hasattr(func, "__self__"):
            ID = f"{func.__self__.__class__.__name__}.{func.__name__}"
        else:
            ID = f"{func.__module__}.{func.__name__}"

        os.makedirs(save_path, exist_ok=True)

        class Profile:

            def __init__(self, name):
                self.name = name
                self.call_count = 0
                self.time_last = 0
                self.time_avg = 0
                self.has_warned = False

            @wraps(func)
            def wrapper(self, *args, **kwargs):

                if save:
                    pr = cProfile.Profile()
                    pr.enable()

                start = perf_counter()

                output = func(*args, **kwargs)

                if save:
                    pr.disable()
                    if self.call_count < max_profs_size:
                        pr.dump_stats(f"{save_path}_call_{self.call_count}.prof")
                    elif not self.has_warned:
                        warnings.warn(
                            f"Max profile size reached ({max_profs_size}). Not saving further profiles."
                        )
                        self.has_warned = True

                end = perf_counter()

                time = end - start
                time_since_last = end - self.time_last

                self.time_avg = (self.time_avg * self.call_count + time) / (
                    self.call_count + 1
                )
                self.call_count += 1

                if verbose:

                    since_last_str = (
                        f"   since last = {time_since_last:0.3f}s)"
                        if self.time_last
                        else ""
                    )
                    ps = f"Call {self.call_count}   time: {time:0.3f}s    avg = {self.time_avg:0.3f}s){since_last_str}"
                    print(f"Profiling {self.name}:: {ps}")

                self.time_last = end

                return output

        return Profile(ID).wrapper

    return decorator
