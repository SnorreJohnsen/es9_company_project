import math

from scipy.stats import linregress
from collections import deque

from uwb_positioning.config import VelocityEstimatorConfig
from uwb_positioning.models import PositionMeasurement


class VelocityEstimator:
    def __init__(self, config: VelocityEstimatorConfig):
        self.config = config
        self.buffer: deque[PositionMeasurement] = deque(maxlen=config.window_length)

    def linear_estimate(self, 
                        measurement: PositionMeasurement,
                        ) -> PositionMeasurement:
        self.buffer.append(measurement) 
        
        vx = math.nan 
        vy = math.nan 
        vz = math.nan

        std_vx = math.nan
        std_vy = math.nan
        std_vz = math.nan

        if len(self.buffer) >= 2:
            first_timestamp_us = self.buffer[0].timestamp_us

            times_s = [(sample.timestamp_us - first_timestamp_us) / 1_000_000 for sample in self.buffer]

            x_result = linregress(times_s, [sample.x for sample in self.buffer])

            y_result = linregress(times_s, [sample.y for sample in self.buffer])

            z_result = linregress(times_s, [sample.z for sample in self.buffer])

            vx = x_result.slope # type: ignore[attr-defined]
            vy = y_result.slope # type: ignore[attr-defined]
            vz = z_result.slope # type: ignore[attr-defined]

            if len(self.buffer) >= 3:
                std_vx = x_result.stderr # type: ignore[attr-defined]
                std_vy = y_result.stderr # type: ignore[attr-defined]
                std_vz = z_result.stderr # type: ignore[attr-defined]

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=measurement.x,
                                   y=measurement.y,
                                   z=measurement.z,
                                   vx=vx,
                                   vy=vy,
                                   vz=vz,
                                   frame=measurement.frame,
                                   std_x=measurement.std_x,
                                   std_y=measurement.std_y,
                                   std_z=measurement.std_z,
                                   std_vx=std_vx,
                                   std_vy=std_vy,
                                   std_vz=std_vz,
                                   valid=measurement.valid,
                                   rejection_reason=measurement.rejection_reason,
                                   )
