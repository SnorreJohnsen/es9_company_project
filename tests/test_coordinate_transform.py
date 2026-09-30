import math
import pytest

from uwb_positioning.config import TransformConfig
from uwb_positioning.models import PositionMeasurement
from uwb_positioning.processing.coordinate_transform import CoordinateTransformer


@pytest.mark.parametrize("input_coordinates, expected_coordinates",
                         [
                             ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                             ((1.0, 2.0, 3.0), (1.0, -2.0, -3.0)),
                             ((-4.0, 5.0, -6.0), (-4.0, -5.0, 6.0)),
                         ],
                         )
def test_transform_coordinates(input_coordinates: tuple[float, float, float],
                               expected_coordinates: tuple[float, float, float],
                               ) -> None:
    config = TransformConfig(origin_x=0.0,
                             origin_y=0.0,
                             origin_z=0.0,
                             local_x_heading_deg=0.0,
                             )
    transformer = CoordinateTransformer(config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=input_coordinates[0],
                                      y=input_coordinates[1],
                                      z=input_coordinates[2],
                                      frame="UWB",
                                      )

    result = transformer.transform(measurement)

    assert result.x == pytest.approx(expected_coordinates[0], abs=1e-9)
    assert result.y == pytest.approx(expected_coordinates[1], abs=1e-9)
    assert result.z == pytest.approx(expected_coordinates[2], abs=1e-9)
    assert result.frame == "NED"

def test_transform_applies_origin_translation() -> None:
    config = TransformConfig(origin_x=10.0,
                             origin_y=20.0,
                             origin_z=5.0,
                             local_x_heading_deg=0.0,
                             )
    transformer = CoordinateTransformer(config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=11.0,
                                      y=22.0,
                                      z=8.0,
                                      frame="UWB",
                                      )

    result = transformer.transform(measurement)

    assert result.x == pytest.approx(1.0, abs=1e-9)
    assert result.y == pytest.approx(-2.0, abs=1e-9)
    assert result.z == pytest.approx(-3.0, abs=1e-9)

def test_transform_applies_heading_rotation() -> None:
    config = TransformConfig(origin_x=0.0,
                             origin_y=0.0,
                             origin_z=0.0,
                             local_x_heading_deg=90.0,
                             )
    transformer = CoordinateTransformer(config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=1.0,
                                      y=2.0,
                                      z=3.0,
                                      frame="UWB",
                                      )

    result = transformer.transform(measurement)

    assert result.x == pytest.approx(2.0, abs=1e-9)
    assert result.y == pytest.approx(1.0, abs=1e-9)
    assert result.z == pytest.approx(-3.0, abs=1e-9)

def test_transform_preserves_measurement_metadata() -> None:
    config = TransformConfig(origin_x=0.0,
                             origin_y=0.0,
                             origin_z=0.0,
                             local_x_heading_deg=0.0,
                             )
    transformer = CoordinateTransformer(config)

    measurement = PositionMeasurement(timestamp_us=123456,
                                      x=1.0,
                                      y=2.0,
                                      z=3.0,
                                      frame="UWB",
                                      std_x=0.1,
                                      std_y=0.2,
                                      std_z=0.3,
                                      valid=False,
                                      rejection_reason="Test rejection",
                                      )

    result = transformer.transform(measurement)

    assert result.timestamp_us == measurement.timestamp_us
    assert result.std_x == measurement.std_x
    assert result.std_y == measurement.std_y
    assert result.std_z == measurement.std_z
    assert result.valid == measurement.valid
    assert result.rejection_reason == measurement.rejection_reason

    assert result.frame == "NED"
