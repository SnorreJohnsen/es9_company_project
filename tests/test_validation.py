import pytest

from uwb_positioning.config import ValidationConfig
from uwb_positioning.models import PositionMeasurement
from uwb_positioning.processing.validation import MeasurementValidator

def make_measurement(timestamp_us: int, 
                     x: float,
                     y: float,
                     z: float,
                     ) -> PositionMeasurement:
    return PositionMeasurement(timestamp_us=timestamp_us,
                                  x=x,
                                  y=y,
                                  z=z,
                                  frame="NED",
                                  )

def test_valid_measurement_is_accepted() -> None:
    config = ValidationConfig()
    validator = MeasurementValidator(config)

    measurement = make_measurement(timestamp_us=1_000_000,
                                   x=1.0,
                                   y=1.0,
                                   z=-1.0,
                                   )
    result = validator.validate(measurement)

    assert result.valid
    assert result.rejection_reason is None

@pytest.mark.parametrize("input_coordinates",
                         [
                             (20.0, 0.0, 0.0),
                             (0.0, 20.0, 0.0),
                             (0.0, 0.0, 20.0),
                         ],
                         )
def test_position_outside_area_is_rejected(input_coordinates: tuple[float, float, float]) -> None:
    config = ValidationConfig(x_min=-5.0,
                              x_max=5.0,
                              y_min=-5.0,
                              y_max=5.0,
                              z_min=-5.0,
                              z_max=5.0,
                              )
    validator = MeasurementValidator(config)

    measurement = make_measurement(timestamp_us=1_000_000,
                                   x=input_coordinates[0],
                                   y=input_coordinates[1],
                                   z=input_coordinates[2],
                                   )
    result = validator.validate(measurement)

    assert not result.valid
    assert result.rejection_reason is not None
                                   
def test_old_timestamp_is_rejected() -> None:
    config = ValidationConfig()
    validator = MeasurementValidator(config)

    first = make_measurement(timestamp_us=2_000_000,
                             x=0.0,
                             y=0.0,
                             z=0.0,
                             )
    second = make_measurement(timestamp_us=1_000_000,
                             x=0.0,
                             y=0.0,
                             z=0.0,
                             )
    validator.validate(first)
    result = validator.validate(second)

    assert not result.valid
    assert result.rejection_reason is not None
    assert "Timestamp is not newer than previous measurement" in result.rejection_reason

def test_imposible_velocity_is_rejected() -> None:
    config = ValidationConfig(x_max=100.0,
                              maximum_velocity=5.0,
                              )
    validator = MeasurementValidator(config)

    first = make_measurement(timestamp_us=1_000_000,
                             x=0.0,
                             y=0.0,
                             z=0.0,
                             )
    second = make_measurement(timestamp_us=1_100_000,
                             x=20.0,
                             y=0.0,
                             z=0.0,
                             )
    validator.validate(first)
    result = validator.validate(second)

    assert not result.valid
    assert result.rejection_reason is not None
    assert "Velocity exceeds configured limit" in result.rejection_reason
