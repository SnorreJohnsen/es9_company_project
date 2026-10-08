import time
import random

from uwb_positioning.config import SimulatorConfig
from uwb_positioning.models import PositionMeasurement
from uwb_positioning.providers.base import PositionProvider
from uwb_positioning.providers.trajectories import calculate_circle_position

class SimulatedProvider(PositionProvider):
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.start_time: float | None = None

        self.random_generator = random.Random(self.config.random_seed)

    def connect(self) -> None:
        self.start_time = time.monotonic()
        print("Simulated position provider connected")
        print("Trajectory: circle")
        print(f"Speed: {self.config.speed_mps} m/s")

        if self.config.duration_s is not None:
            print(f"Simultion duration: {self.config.duration_s}")

    def get_measurement(self) -> PositionMeasurement:
        if self.start_time is None:
            raise RuntimeError("Provider is not connected")

        elapsed_time = time.monotonic() - self.start_time

        x, y, z = calculate_circle_position(elapsed_time=elapsed_time,
                                            speed=self.config.speed_mps,
                                            config=self.config.trajectory,
                                            )

        x, y, z = self._add_noise(x, y, z)

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

    def _add_noise(self,
                   x: float,
                   y: float,
                   z: float,
                   ) -> tuple[float, float, float]:
        noise_std = self.config.noise_std

        return (x + self.random_generator.gauss(0.0, noise_std),
                y + self.random_generator.gauss(0.0, noise_std),
                z + self.random_generator.gauss(0.0, noise_std),
                )

    def disconnect(self) -> None:
        self.start_time = None
        print("Simulated position provider disconnected")
