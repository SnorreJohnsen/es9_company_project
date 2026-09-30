from abc import ABC, abstractmethod

from uwb_positioning.models import PositionMeasurement


class PositionProvider(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Connect to the positioning system or simulator"""
        ...

    @abstractmethod
    def get_measurement(self) -> PositionMeasurement:
        """Return the next position measurement"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the positioning system or simulator"""
        ...
