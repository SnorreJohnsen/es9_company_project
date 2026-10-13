import random

from uwb_positioning.models import PositionMeasurement
from uwb_positioning.config import SitlSimulationConfig


class SitlPositionNoise:
    def __init__(self, config: SitlSimulationConfig):
        self.config = config
        self.random_generator = random.Random(config.random_seed)

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
                                   frame=measurement.frame,
                                   std_x=self.config.horizontal_noise_std_m,
                                   std_y=self.config.horizontal_noise_std_m,
                                   std_z=self.config.horizontal_noise_std_m,
                                   )
