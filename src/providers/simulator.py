import time
import math
import random

from config import SimulatorConfig
from models import PositionMeasurement
from providers.base import PositionProvider

class SimulatedProvider(PositionProvider):
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.start_time: float | None = None

    def connect(self) -> None:
        self.start_time = time.monotonic()
        print("Simulated position provider connected")

    def get_measurement(self) -> PositionMeasurement:
        if self.start_time is None:
            raise RuntimeError("Provider is not connected")

        elapsed_time = time.monotonic() - self.start_time

        angle = self.config.angular_speed * elapsed_time

        x = self.config.radius * math.cos(angle)
        y = self.config.radius * math.sin(angle)
        z = self.config.altitude

        x += random.gauss(0.0, self.config.noise_std)
        y += random.gauss(0.0, self.config.noise_std)
        z += random.gauss(0.0, self.config.noise_std)

        return PositionMeasurement(
                timestamp_us=time.monotonic_ns() // 1000,
                x=x,
                y=y,
                z=z,
                frame="UWB",
                std_x=self.config.noise_std,
                std_y=self.config.noise_std,
                std_z=self.config.noise_std,
                )

    def disconnect(self) -> None:
        self.start_time = None
        print("Simulated position provider disconnected")
