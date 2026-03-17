import abc

GRIPPER_OPEN = 0
GRIPPER_CLOSED = 1


class BaseGripper(abc.ABC):


    open_raw: float
    closed_raw: float

    @abc.abstractmethod
    def connect(self) -> None:
        """Initialize / connect to the gripper hardware."""

    @abc.abstractmethod
    def get_position(self) -> float:
        """Return normalized gripper position in [0.0, 1.0] (0=open, 1=closed)."""

    @abc.abstractmethod
    def set_position(self, pos: float, wait: bool = False) -> None:
        """Command a normalized position in [0.0, 1.0]."""

    def _to_raw(self, pos: float) -> float:
        """Convert a normalized [0.0, 1.0] position to a raw hardware value."""
        return self.open_raw + pos * (self.closed_raw - self.open_raw)

    def _to_normalized(self, raw: float) -> float:
        """Convert a raw hardware value to a normalized [0.0, 1.0] position."""
        span = self.closed_raw - self.open_raw
        if span == 0:
            return 0.0
        return (raw - self.open_raw) / span

