import math

from uwb_positioning.config import LocalTransformConfig, GeographicTransformConfig
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

class NedToLocalFluTransformer:
    def __init__(self, config: LocalTransformConfig):
        self.config = config

    def transform(self,
                  measurement: PositionMeasurement,
                  ) -> PositionMeasurement:
        heading_rad = math.radians(self.config.local_x_heading_deg)

        # A^-1 = A for this transformation
        local_x = (math.cos(heading_rad) * measurement.x + math.sin(heading_rad) * measurement.y)

        local_y = (math.sin(heading_rad) * measurement.x - math.cos(heading_rad) * measurement.y)

        local_z = -measurement.z

        # translate drone origin back to UWB origin
        x = local_x + self.config.origin_x
        y = local_y + self.config.origin_y
        z = local_z + self.config.origin_z

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=x,
                                   y=y,
                                   z=z,
                                   frame="FLU",
                                   std_x=measurement.std_x,
                                   std_y=measurement.std_y,
                                   std_z=measurement.std_z,
                                   valid=measurement.valid,
                                   rejection_reason=measurement.rejection_reason,
                                   )

class NedToGeographicTransformer:
    def __init__(self, config: GeographicTransformConfig):
        self.config = config

    def transform(self,
                  measurement: PositionMeasurement,
                  ) -> GeographicMeasurement:
        """
        Transforms a drone NED coordinate system to geographic lat, lon and alt.
        Assumes std_x == std_y.
        """
        latitude0_rad = math.radians(self.config.origin_latitude_deg)

        latitude_deg = (self.config.origin_latitude_deg
                       + math.degrees(measurement.x / EARTH_RADIUS_M)
                        )

        longitude_deg = (self.config.origin_longitude_deg
                         + math.degrees(measurement.y / (EARTH_RADIUS_M * math.cos(latitude0_rad)))
                         )
                        
        longitude_deg = ((longitude_deg + 180.0) % 360.0) - 180

        altitude_m = self.config.origin_altitude_m - measurement.z

        return GeographicMeasurement(timestamp_us=measurement.timestamp_us,
                                     latitude_deg=latitude_deg,
                                     longitude_deg=longitude_deg,
                                     altitude_m=altitude_m,
                                     std_horizontal_m=measurement.std_x,
                                     std_vertical_m=measurement.std_z,
                                     )

class GeographicToNedTransformer:
    def __init__(self, config: GeographicTransformConfig):
        self.config = config

    def transform(self,
                  measurement: GeographicMeasurement,
                  ) -> PositionMeasurement:
        latitude0_rad = math.radians(self.config.origin_latitude_deg)

        delta_latitude_rad = math.radians(measurement.latitude_deg - self.config.origin_latitude_deg)

        # normalize longitude difference to [-180, 180) degrees
        delta_longitude_deg = ((measurement.longitude_deg - self.config.origin_longitude_deg + 180.0)
                               % 360.0) - 180.0

        delta_longitude_rad = math.radians(delta_longitude_deg)

        # local approximations with arc lenght formula s=radius*Theta. Suitable for small testing areas.
        north_m = EARTH_RADIUS_M * delta_latitude_rad

        east_m = EARTH_RADIUS_M * math.cos(latitude0_rad) * delta_longitude_rad

        down_m = self.config.origin_altitude_m - measurement.altitude_m

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=north_m,
                                   y=east_m,
                                   z=down_m,
                                   frame="NED",
                                   std_x=measurement.std_horizontal_m,
                                   std_y=measurement.std_horizontal_m,
                                   std_z=measurement.std_vertical_m
                                   )

class SitlGeographicToNedTransformer:
    def __init__(self):
        # initial fixed origin
        self._origin: tuple[float, float, float] | None = None # [latitude, longitude, altitude]

    def transform(self,
                  measurement: GeographicMeasurement,
                  ) -> PositionMeasurement:
        """ 
        Transforms geographic lat, lon, alt to NED coordinates.
        """

        latitude_deg = measurement.latitude_deg
        longitude_deg = measurement.longitude_deg
        altitude_m = measurement.altitude_m

        values = (latitude_deg,
                  longitude_deg,
                  altitude_m,
                  )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cannot transform non-finite geographic coordinates")

        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"Invalid latitude: {latitude_deg}")

        if not -180.0 <= longitude_deg <= 180.0:
            raise ValueError(f"Invalid longitude: {longitude_deg}")

        # save initial position as origin. should not be changed during simulation
        if self._origin is None:
            self._origin = (latitude_deg,
                            longitude_deg,
                            altitude_m,
                            )

        latitude0_deg, longitude0_deg, altitude0_m = self._origin

        latitude0_rad = math.radians(latitude0_deg)

        delta_latitude_rad = math.radians(latitude_deg - latitude0_deg)

        # normalize longitude difference to [-180, 180) degrees
        delta_longitude_deg = (
                (longitude_deg - longitude0_deg + 180.0) % 360.0
                ) - 180.0

        delta_longitude_rad = math.radians(delta_longitude_deg)


        # local approximations with arc lenght formula s=radius*Theta. Suitable for small testing areas.
        north_m = (EARTH_RADIUS_M * delta_latitude_rad)

        east_m = (EARTH_RADIUS_M
                  * math.cos(latitude0_rad)
                  * delta_longitude_rad
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

