from timeit import default_timer as timer
from numpy import mean


class SimpleFPSCounter:

    def __init__(self, length=5):
        self.last_time = timer()
        self.last_intervals = [0] * length
        self.length = length
        self.counter = 0

    def update(self):
        new_time = timer()
        self.last_intervals[self.counter] = new_time - self.last_time
        self.last_time = new_time
        if self.counter < self.length - 1:
            self.counter += 1
        else:
            self.counter = 0
        return 1 / mean(self.last_intervals)


class SimpleFPSLimiter:

    def __init__(self, fps):
        self.fps = fps
        self.max_delay = 1000 / fps
        self._start_time = timer()
        self._end_time = timer()

    def start(self):
        self._start_time = timer()

    def end(self):
        self._end_time = timer()

    def delay(self):
        between = float(self._end_time - self._start_time) * 1000
        return max(int(self.max_delay - between), 1)