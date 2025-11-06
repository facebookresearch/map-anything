# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for timing code blocks
"""

import time
from contextlib import ContextDecorator

import numpy as np


class BlockTimeManager:
    """
    Manages a collection of timers and their formatting options.

    This class serves as a central registry for Timer objects, allowing them to be
    accessed by name and maintaining their formatting preferences.

    Attributes:
        timers (dict): Dictionary mapping timer names to Timer objects
        timer_fmts (dict): Dictionary mapping timer names to their display formats
        window_size (int): Default window size for calculating windowed averages
        buf_size (int): Default buffer size for storing timing measurements
    """

    def __init__(self, window_size=10, buf_size=100000):
        self.timers = dict()
        self.timer_fmts = dict()
        self.window_size = window_size
        self.buf_size = buf_size


btm = BlockTimeManager(window_size=100000)


class Timer:
    """
    Core timing class that tracks execution times.

    This class provides the fundamental timing functionality, storing timing measurements
    and calculating various statistics.

    Attributes:
        name (str): Identifier for this timer
        buf_size (int): Maximum number of timing measurements to store
        window_size (int): Number of most recent measurements to use for windowed statistics
        measures_arr (numpy.ndarray): Array storing start and end times of measurements
        current_start (float or None): Start time of current measurement
        current_end (float or None): End time of current measurement
    """

    def __init__(self, name, window_size, buf_size=100000):
        self.name = name
        self.buf_size = buf_size
        self.window_size = window_size
        self.init()

    def init(self):
        """Initialize or reset the timer's state."""
        self.measures_arr = np.empty((0, 2))  # LIFO
        self.current_start = None
        self.current_end = None

    def reset(self):
        """Reset the timer to its initial state."""
        self.init()

    def tic(self):
        """Start a new timing measurement."""
        if self.current_start is not None:
            # another tic executed before a toc
            self.toc()
        self.current_start = time.perf_counter()

    def toc(self):
        """End the current timing measurement."""
        self.current_end = time.perf_counter()
        self._add_current_measure()

    def _add_current_measure(self):
        """Add the current timing measurement to the measurements array."""
        self.measures_arr = np.concatenate(
            [
                np.array([[self.current_start, self.current_end]]),
                self.measures_arr[: self.buf_size],
            ]
        )
        self.current_start = None
        self.current_end = None

    @property
    def avg(self) -> float:
        """Calculate the average execution time across all measurements."""
        return np.mean(self.measures_arr[:, 1] - self.measures_arr[:, 0])

    @property
    def wavg(self) -> float:
        """Calculate the windowed average execution time using the most recent measurements."""
        return np.mean(
            self.measures_arr[: self.window_size, 1]
            - self.measures_arr[: self.window_size, 0]
        )

    @property
    def max(self) -> float:
        """Return the maximum execution time."""
        return np.max(self.measures_arr[:, 1] - self.measures_arr[:, 0])

    @property
    def min(self) -> float:
        """Return the minimum execution time."""
        return np.min(self.measures_arr[:, 1] - self.measures_arr[:, 0])

    @property
    def total(self) -> float:
        """Return the total execution time across all measurements."""
        return np.sum(self.measures_arr[:, 1] - self.measures_arr[:, 0])

    @property
    def latest(self) -> float:
        """Return the most recent execution time."""
        return self.measures_arr[0, 1] - self.measures_arr[0, 0]

    @property
    def median(self) -> float:
        """Return the median execution time."""
        return np.median(self.measures_arr[:, 1] - self.measures_arr[:, 0])

    @property
    def var(self) -> float:
        """Return the variance of execution times."""
        return np.var(self.measures_arr[:, 1] - self.measures_arr[:, 0])


class BlockTimer(ContextDecorator):
    """
    A context manager and decorator for timing code blocks.

    This class provides a convenient interface for timing code execution, either as a
    context manager (with statement) or as a decorator. It uses the Timer class for
    the actual timing functionality.

    Attributes:
        name (str): Identifier for this timer
        fmt (str or None): Format string for displaying timing information
        timer (Timer): The underlying Timer object
        num_calls (int): Number of times this timer has been called
    """

    @staticmethod
    def timers():
        """Return a list of all registered timer names."""
        return list(btm.timers.keys())

    def __init__(self, name, fmt=None, window_size=100):
        self.name = name
        if name in btm.timers:
            self.timer = btm.timers[name]
            # restore format
            self.fmt = fmt if fmt is not None else btm.timer_fmts[name]
        else:
            self.timer = Timer(name, btm.window_size, btm.buf_size)
            btm.timers[name] = self.timer
            btm.timer_fmts[name] = fmt
        self.timer.window_size = window_size
        self._default_fmt = "[{name}] num: {num} latest: {latest:.4f} --wind_avg: {wavg:.4f} -- avg: {avg:.4f} --var: {var:.4f} -- total: {total:.4f}"
        if fmt == "default":
            self.fmt = self._default_fmt
        # extend here for new formats
        else:
            self.fmt = None

        self.num_calls = 0

    def __enter__(self) -> "Timer":
        """Start timing when entering a context."""
        self.tic()
        return self

    def __exit__(self, *args):
        """End timing when exiting a context and optionally display results."""
        self.toc()
        if self.fmt is not None:
            print(str(self))

    def __str__(self) -> str:
        """Return a string representation of the timer."""
        return self.display()

    def reset(self):
        """Reset the timer and call counter."""
        self.timer.reset()
        self.num_calls = 0

    def display(self, fmt=None):
        """
        Format and return timing information.

        Args:
            fmt (str, optional): Format string to use. If None, uses the timer's format.

        Returns:
            str: Formatted timing information
        """
        if fmt is None:
            if self.fmt is not None:
                fmt = self.fmt
            else:
                fmt = self._default_fmt
        return fmt.format(
            name=self.name,
            num=self.num_calls,
            latest=self.latest,
            wavg=self.wavg,
            avg=self.avg,
            var=self.var,
            total=self.total,
        )

    def tic(self):
        """Start a new timing measurement and increment the call counter."""
        self.timer.tic()
        self.num_calls += 1

    def toc(self, display=False):
        """
        End the current timing measurement.

        Args:
            display (bool): Whether to return a formatted display string

        Returns:
            str or None: Formatted timing information if display is True
        """
        self.timer.toc()
        if display:
            return self.display()

    @property
    def latest(self) -> float:
        """Return the most recent execution time."""
        return self.timer.latest

    @property
    def avg(self) -> float:
        """Return the average execution time."""
        return self.timer.avg

    @property
    def wavg(self) -> float:
        """Return the windowed average execution time."""
        return self.timer.wavg

    @property
    def max(self) -> float:
        """Return the maximum execution time."""
        return self.timer.max

    @property
    def min(self) -> float:
        """Return the minimum execution time."""
        return self.timer.min

    @property
    def total(self) -> float:
        """Return the total execution time."""
        return self.timer.total

    @property
    def median(self) -> float:
        """Return the median execution time."""
        return self.timer.median

    @property
    def var(self) -> float:
        """Return the variance of execution times."""
        return self.timer.var


if __name__ == "__main__":

    @BlockTimer("fct", "default")
    def fct(bobo):
        time.sleep(0.5)

    fct(2)

    for i in range(10):
        with BlockTimer("affe", "default"):
            time.sleep(0.1)
    for i in range(1000):
        with BlockTimer("test", None):
            time.sleep(0.001)

        # BlockTimer("test").display = f"""avg: {BlockTimer("test").avg}  total: {BlockTimer("test").total}"""
        # print(str(BlockTimer("test")))

    print(BlockTimer("test"))
    BlockTimer("test").tic()
    BlockTimer("t2", "default").tic()
    time.sleep(0.4)
    print(BlockTimer("t2").toc(True))

    time.sleep(0.4)
    print(BlockTimer("test").toc(True))
