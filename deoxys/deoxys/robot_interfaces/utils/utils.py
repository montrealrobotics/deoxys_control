import time


class Rate:

    def __init__(self, *, duration: float):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration: float = None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()

