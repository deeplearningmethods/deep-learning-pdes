# timer.py

'''
Implements a custom timer class

'''


import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self.total_elapsed_time = 0
        self.last_start = None
        self.counting = False

    def start(self):
        """Start a new timer"""
        if self.counting:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self.counting = True
        self.last_start = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if not self.counting:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self.last_start
        self.total_elapsed_time  = self.total_elapsed_time + elapsed_time
        self.counting = False

        return elapsed_time

    def read_time(self):
        if self.counting:
            raise TimerError(f"Timer is running. Use .stop() to stop it and then read the time")
        return self.total_elapsed_time

    def reset(self):
        """Reset the timer to 0"""
        self.total_elapsed_time = 0
        self.last_start = None
        self.counting = False
