import math
import pytest

from uwb_positioning.config import GeographicTransformConfig
from uwb_positioning.models import GeographicMeasurement, PositionMeasurement
from uwb_positioning.processing.coordinate_transform import EARTH_RADIUS_M, GeographicToNedTransformer, NedToGeographicTransformer

@pytest.fixture
def geographic_config() -> GeographicTransformConfig:
    return GeographicTransformConfig(origin_latitude_deg=55.0,
                                     origin_longitude_deg=10.0,
                                     origin_altitude_m=100.0,
                                     )

def test_geographic_origin_to_ned_is_zero(geographic_config: GeographicTransformConfig) -> None:
    transformer = GeographicToNedTransformer(geographic_config)

    measurement = GeographicMeasurement(timestamp_us=1000,
                                        latitude_deg=55.0,
                                        longitude_deg=10.0,
                                        altitude_m=100.0,
                                        )
    result = transformer.transform(measurement)

    assert result.x == pytest.approx(0.0, abs=1e-9)
    assert result.y == pytest.approx(0.0, abs=1e-9)
    assert result.z == pytest.approx(0.0, abs=1e-9)

def test_ned_zero_to_geographic_origin(geographic_config: GeographicTransformConfig) -> None:
    transformer = NedToGeographicTransformer(geographic_config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=0.0,
                                      y=0.0,
                                      z=0.0,
                                      frame="NED",
                                      )
    result = transformer.transform(measurement)

    assert result.latitude_deg == pytest.approx(geographic_config.origin_latitude_deg, abs=1e-9)
    assert result.longitude_deg == pytest.approx(geographic_config.origin_longitude_deg, abs=1e-9)
    assert result.altitude_m == pytest.approx(geographic_config.origin_altitude_m, abs=1e-9)

def test_geographic_to_ned_coordinates(geographic_config: GeographicTransformConfig) -> None:
    transformer = GeographicToNedTransformer(geographic_config)

    north_m = 10.0
    east_m = 20.0
    down_m = -5.0

    latitude_offset_deg = math.degrees(north_m / EARTH_RADIUS_M)

    origin_latitude_rad = math.radians(geographic_config.origin_latitude_deg)
    longitude_offset_deg = math.degrees(east_m / 
                                        (EARTH_RADIUS_M * math.cos(origin_latitude_rad))
                                        )

    measurement = GeographicMeasurement(timestamp_us=1000,
                                        latitude_deg=geographic_config.origin_latitude_deg + latitude_offset_deg,
                                        longitude_deg=geographic_config.origin_longitude_deg + longitude_offset_deg,
                                        altitude_m=geographic_config.origin_altitude_m - down_m,
                                        )
    result = transformer.transform(measurement)

    assert result.x == pytest.approx(north_m, abs=1e-9)
    assert result.y == pytest.approx(east_m, abs=1e-9)
    assert result.z == pytest.approx(down_m, abs=1e-9)
    assert result.frame == "NED"

def test_ned_to_geographic_coordinates(geographic_config: GeographicTransformConfig) -> None:
    transformer = NedToGeographicTransformer(geographic_config)

    north_m = 10.0
    east_m = 20.0
    down_m = -5.0

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=north_m,
                                      y=east_m,
                                      z=down_m,
                                      frame="NED",
                                      )
    result = transformer.transform(measurement)

    expected_latitude_deg = geographic_config.origin_latitude_deg + math.degrees(north_m / EARTH_RADIUS_M)

    origin_latitude_rad = math.radians(geographic_config.origin_latitude_deg)
    expected_longitude_deg = geographic_config.origin_longitude_deg + math.degrees(east_m
                                                                                   / (EARTH_RADIUS_M * math.cos(origin_latitude_rad))
                                                                                   )
    expected_altitude_m = geographic_config.origin_altitude_m - down_m

    assert result.latitude_deg == pytest.approx(expected_latitude_deg, abs=1e-9)
    assert result.longitude_deg == pytest.approx(expected_longitude_deg, abs=1e-9)
    assert result.altitude_m == pytest.approx(expected_altitude_m, abs=1e-9)

@pytest.mark.parametrize("north_m, east_m, down_m",
                         [(0.0, 0.0, 0.0),
                          (10.0, 20.0, -5.0),
                          (-10.0, -20.0, 5.0),
                          (100.0, -50.0, -25.0),
                          ]
                         )
def test_ned_and_geographic_transformers_are_inverse(geographic_config: GeographicTransformConfig,
                                                     north_m: float,
                                                     east_m: float,
                                                     down_m: float,
                                                     ) -> None:
    ned_to_geographic = NedToGeographicTransformer(geographic_config)
    geographic_to_ned = GeographicToNedTransformer(geographic_config)

    original = PositionMeasurement(timestamp_us=1000,
                                   x=north_m,
                                   y=east_m,
                                   z=down_m,
                                   frame="NED",
                                   std_x=0.1,
                                   std_y=0.1,
                                   std_z=0.2,
                                   )
    geographic = ned_to_geographic.transform(original)
    result = geographic_to_ned.transform(geographic)

    assert result.x == pytest.approx(original.x, abs=1e-9)
    assert result.y == pytest.approx(original.y, abs=1e-9)
    assert result.z == pytest.approx(original.z, abs=1e-9)
    assert result.frame == "NED"

    assert result.timestamp_us == original.timestamp_us
    assert result.std_x == original.std_x
    assert result.std_y == original.std_y
    assert result.std_z == original.std_z

def test_geographic_and_ned_transformers_preserve_uncertainty(geographic_config: GeographicTransformConfig) -> None:
    geographic_to_ned = GeographicToNedTransformer(geographic_config)

    geographic_measurement = GeographicMeasurement(timestamp_us=1000,
                                                   latitude_deg=55.0,
                                                   longitude_deg=10.0,
                                                   altitude_m=100.0,
                                                   std_horizontal_m=0.1,
                                                   std_vertical_m=0.2,
                                                   )
    ned_to_geographic = NedToGeographicTransformer(geographic_config)

    ned_measurement = PositionMeasurement(timestamp_us=2000,
                                          x=10.0,
                                          y=20.0,
                                          z=-5.0,
                                          frame="NED",
                                          std_x=0.3,
                                          std_y=0.3,
                                          std_z=0.4,
                                          )
    geographic_result = ned_to_geographic.transform(ned_measurement)

    assert geographic_result.timestamp_us == ned_measurement.timestamp_us
    assert geographic_result.std_horizontal_m == ned_measurement.std_x
    assert geographic_result.std_horizontal_m == ned_measurement.std_y
    assert geographic_result.std_vertical_m == ned_measurement.std_z

                                                                               

