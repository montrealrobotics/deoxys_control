import time

class Rate:
    def __init__(self, *, duration: float):
        self.duration = duration
        self.last = time.monotonic()

    def sleep(self, duration: float = None) -> None:
        duration = self.duration if duration is None else duration

        if duration < 0:
            raise ValueError("Duration must be non-negative")

        now = time.monotonic()
        passed = now - self.last
        remaining = duration - passed

        if remaining > 0.0001:
            time.sleep(remaining)

        self.last = time.monotonic()