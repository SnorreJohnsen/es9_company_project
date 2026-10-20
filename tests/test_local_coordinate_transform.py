import pytest

from uwb_positioning.config import LocalTransformConfig
from uwb_positioning.models import PositionMeasurement
from uwb_positioning.processing.coordinate_transform import LocalFluToNedTransformer, NedToLocalFluTransformer


@pytest.fixture
def zero_heading_config() -> LocalTransformConfig:
    return LocalTransformConfig(origin_x=0.0, 
                                origin_y=0.0, 
                                origin_z=0.0, 
                                local_x_heading_deg=0.0,
                                )

@pytest.mark.parametrize("input_coordinates, expected_coordinates",
                         [
                             ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                             ((1.0, 2.0, 3.0), (1.0, -2.0, -3.0)),
                             ((-4.0, 5.0, -6.0), (-4.0, -5.0, 6.0)),
                         ],
                         )
def test_local_flu_to_ned_coordinates(zero_heading_config: LocalTransformConfig, 
                                      input_coordinates: tuple[float, float, float], 
                                      expected_coordinates: tuple[float, float, float],
                                      ) -> None:
    transformer = LocalFluToNedTransformer(zero_heading_config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=input_coordinates[0],
                                      y=input_coordinates[1],
                                      z=input_coordinates[2],
                                      frame="FLU",
                                      )

    result = transformer.transform(measurement)

    assert result.x == pytest.approx(expected_coordinates[0], abs=1e-9)
    assert result.y == pytest.approx(expected_coordinates[1], abs=1e-9)
    assert result.z == pytest.approx(expected_coordinates[2], abs=1e-9)
    assert result.frame == "NED"

@pytest.mark.parametrize("input_coordinates, expected_coordinates",
                         [
                             ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                             ((1.0, 2.0, 3.0), (1.0, -2.0, -3.0)),
                             ((-4.0, 5.0, -6.0), (-4.0, -5.0, 6.0)),
                         ],
                         )
def test_transform_ned_to_local_flu_coordinates(zero_heading_config: LocalTransformConfig,
                                                input_coordinates: tuple[float, float, float], 
                                                expected_coordinates: tuple[float, float, float],
                                                ) -> None:
    transformer = NedToLocalFluTransformer(zero_heading_config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=input_coordinates[0],
                                      y=input_coordinates[1],
                                      z=input_coordinates[2],
                                      frame="FLU",
                                      )
    
    result = transformer.transform(measurement)

    assert result.x == pytest.approx(expected_coordinates[0], abs=1e-9)
    assert result.y == pytest.approx(expected_coordinates[1], abs=1e-9)
    assert result.z == pytest.approx(expected_coordinates[2], abs=1e-9)
    assert result.frame == "FLU"


def test_local_flu_to_ned_applies_origin_and_heading() -> None:
    config = LocalTransformConfig(origin_x=10.0, 
                                  origin_y=20.0, 
                                  origin_z=5.0, 
                                  local_x_heading_deg=90.0,
                                  )
    transformer = LocalFluToNedTransformer(config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=11.0,
                                      y=22.0,
                                      z=8.0,
                                      frame="FLU",
                                      )
    result = transformer.transform(measurement)

    assert result.x == pytest.approx(2.0, abs=1e-9)
    assert result.y == pytest.approx(1.0, abs=1e-9)
    assert result.z == pytest.approx(-3.0, abs=1e-9)

def test_ned_to_local_flu_applies_origin_and_heading() -> None:
    config = LocalTransformConfig(origin_x=10.0,
                                  origin_y=20.0,
                                  origin_z=5.0,
                                  local_x_heading_deg=90.0,
                                  )
    transformer = NedToLocalFluTransformer(config)

    measurement = PositionMeasurement(timestamp_us=1000,
                                      x=2.0,
                                      y=1.0,
                                      z=-3.0,
                                      frame="NED",
                                      )
    result = transformer.transform(measurement)

    assert result.x == pytest.approx(11.0, abs=1e-9)
    assert result.y == pytest.approx(22.0, abs=1e-9)
    assert result.z == pytest.approx(8.0, abs=1e-9)

@pytest.mark.parametrize("heading_deg",
                         [0.0, 30.0, 90.0, 180.0, 270.0],
                         )
def test_local_flu_and_ned_transformers_are_inverse(heading_deg: float) -> None:
    config = LocalTransformConfig(origin_x=10.0,
                                  origin_y=20.0,
                                  origin_z=5.0,
                                  local_x_heading_deg=heading_deg,
                                  )
    flu_to_ned = LocalFluToNedTransformer(config)
    ned_to_flu = NedToLocalFluTransformer(config)

    original = PositionMeasurement(timestamp_us=1000,
                                   x=12.5,
                                   y=17.0,
                                   z=8.5,
                                   frame="FLU",
                                   std_x=0.1,
                                   std_y=0.1,
                                   std_z=0.2,
                                   )
    ned = flu_to_ned.transform(original)
    result = ned_to_flu.transform(ned)

    assert result.x == pytest.approx(original.x, abs=1e-9)
    assert result.y == pytest.approx(original.y, abs=1e-9)
    assert result.z == pytest.approx(original.z, abs=1e-9)
    assert result.frame == "FLU"

    assert result.timestamp_us == original.timestamp_us
    assert result.std_x == original.std_x
    assert result.std_y == original.std_y
    assert result.std_z == original.std_z
