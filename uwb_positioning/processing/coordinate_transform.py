import math

from uwb_positioning.config import TransformConfig
from uwb_positioning.models import PositionMeasurement


class CoordinateTransformer:
    def __init__(self, config: TransformConfig):
        self.config = config

    def transform(self, 
                  measurement: PositionMeasurement,
                  ) -> PositionMeasurement:
        """
        Transforms a local right handed x-forward, y-left, z-up coordinate system to NED.
        Assumes std_x == std_y.
        """

        # translate uwb origin to drone origin
        translated_x = measurement.x - self.config.origin_x
        translated_y = measurement.y - self.config.origin_y
        translated_z = measurement.z - self.config.origin_z

        # rotate x heading to match north. x = 0 is north, x = 90 is east
        heading_rad = math.radians(self.config.local_x_heading_deg)

        north = (math.cos(heading_rad) * translated_x + math.sin(heading_rad) * translated_y)
        east = (math.sin(heading_rad) * translated_x - math.cos(heading_rad) * translated_y)
        down = -translated_z

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=north,
                                   y=east,
                                   z=down,
                                   frame="NED",
                                   std_x=measurement.std_x,
                                   std_y=measurement.std_y,
                                   std_z=measurement.std_z,
                                   valid=measurement.valid,
                                   rejection_reason=measurement.rejection_reason,
                                   )
