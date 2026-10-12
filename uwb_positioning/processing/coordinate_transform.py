import math

from uwb_positioning.config import LocalTransformConfig
from uwb_positioning.models import PositionMeasurement, GeographicMeasurement


EARTH_RADIUS_M = 6_378_137.0

class LocalFluToNedTransformer:
    def __init__(self, config: LocalTransformConfig):
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

class GeograhicToNedTransformer:
    def __init__(self):
        # initial fixed origin
        self._origin: tuple[float, float, float] | None = None # [latitude, longtitude, altitude]

    def transform(self,
                  measurement: GeographicMeasurement,
                  ) -> PositionMeasurement:
        """ 
        Transforms geographic lat, lon, alt to NED coordinates.
        """

        latitude_deg = measurement.latitude_deg
        longtitude_deg = measurement.longtitude_deg
        altitude_m = measurement.altitude_m

        values = (latitude_deg,
                  longtitude_deg,
                  altitude_m,
                  )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cannot transform non-finite geographic coordinates")

        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"Invalid latitude: {latitude_deg}")

        if not -180.0 <= longtitude_deg <= 180.0:
            raise ValueError(f"Invalid longtitude: {longtitude_deg}")

        # save initial position as origin. should not be changed during simulation
        if self._origin is None:
            self._origin = (latitude_deg,
                            longtitude_deg,
                            altitude_m,
                            )

        latitude0_deg, longtitude0_deg, altitude0_m = self._origin

        latitude0_rad = math.radians(latitude0_deg)

        delta_latitude_rad = math.radians(latitude_deg - latitude0_deg)

        # normalize longtitude difference to [-180, 180) degrees
        delta_longtitude_deg = (
                (longtitude_deg - longtitude0_deg + 180.0) % 360.0
                ) - 180.0

        delta_longtitude_rad = math.radians(delta_longtitude_deg)


        # local approximations with arc lenght formula s=radius*Theta. Suitable for small testing areas.
        north_m = (EARTH_RADIUS_M * delta_latitude_rad)

        east_m = (EARTH_RADIUS_M
                  * math.cos(latitude0_rad)
                  * delta_longtitude_rad
                  )

        down_m = altitude0_m - altitude_m

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=north_m,
                                   y=east_m,
                                   z=down_m,
                                   frame="NED",
                                   std_x=measurement.std_horizontal_m,
                                   std_y=measurement.std_horizontal_m,
                                   std_z=measurement.std_vertical_m,
                                   )

