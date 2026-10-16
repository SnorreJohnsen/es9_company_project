import random
import time
import math

from collections import deque
from dataclasses import dataclass

from uwb_positioning.models import PositionMeasurement
from uwb_positioning.config import SitlSimulationConfig


@dataclass
class BufferedMeasurement:
    buffered_time_us: int
    release_time_us: int
    measurement: PositionMeasurement


class MeasurementDegradation:
    def __init__(self, config: SitlSimulationConfig):
        self.config = config
        self.random_generator = random.Random(config.random_seed)
        self.delay_buffer: deque[BufferedMeasurement] = deque()

    def add_gaussian_noise(self,
                           measurement: PositionMeasurement,
                           ) -> PositionMeasurement:
        horizontal_noise_std_m = self.config.horizontal_noise_std_m
        vertical_noise_std_m = self.config.vertical_noise_std_m

        noisy_x = measurement.x + self.random_generator.gauss(0.0, horizontal_noise_std_m)
        noisy_y = measurement.y + self.random_generator.gauss(0.0, horizontal_noise_std_m)
        noisy_z = measurement.z + self.random_generator.gauss(0.0, vertical_noise_std_m)

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=noisy_x,
                                   y=noisy_y,
                                   z=noisy_z,
                                   vx=math.nan,
                                   vy=math.nan,
                                   vz=math.nan,
                                   frame=measurement.frame,
                                   std_x=self.config.horizontal_noise_std_m,
                                   std_y=self.config.horizontal_noise_std_m,
                                   std_z=self.config.vertical_noise_std_m,
                                   )

    def buffer_measurement(self,
                           measurement: PositionMeasurement,
                           ) -> None:
        current_time_us = time.monotonic_ns() // 1000
        delay_us = int(self.config.delay_s * 1_000_000)

        self.delay_buffer.append(
                BufferedMeasurement(buffered_time_us=current_time_us,
                                    release_time_us=current_time_us + delay_us, 
                                    measurement=measurement,
                                    )
                )

    def get_buffer_measurement(self) -> PositionMeasurement | None:
        if not self.delay_buffer:
            return None

        current_time_us = time.monotonic_ns() // 1000
        oldest_buffered_measurement = self.delay_buffer[0]

        if current_time_us < oldest_buffered_measurement.release_time_us:
            return None

        return self.delay_buffer.popleft().measurement
