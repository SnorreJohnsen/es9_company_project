import numpy as np
import pytest

from uwb_positioning.config import TrilaterationConfig
from uwb_positioning.models import RangeMeasurement, RangeSetMeasurement
from uwb_positioning.processing.trilateration import TrilaterationSolver


TEST_ANCHORS = {
    0: (0.0, 0.0, 0.0),
    7: (3.0, 0.0, 0.0),
    2: (3.0, 6.0, 0.0),
    5: (0.0, 6.0, 0.0),

    4: (0.0, 0.0, 3.0),
    3: (3.0, 0.0, 3.0),
    6: (3.0, 6.0, 3.0),
    1: (0.0, 6.0, 3.0),
}


@pytest.fixture
def config() -> TrilaterationConfig:
    return TrilaterationConfig(anchors=TEST_ANCHORS.copy(), 
                               minimum_anchors=6,
                               )


def create_range_set(anchors: dict[int, tuple[float, float, float]], 
                     position: tuple[float, float, float], 
                     anchor_ids: tuple[int, ...] = tuple(range(8)), 
                     noise_std_m: float = 0.0,
                     ) -> RangeSetMeasurement:
    """Create synthetic ranges from a known 3D position."""

    rng = np.random.default_rng(42)
    position_array = np.asarray(position, dtype=float)
    ranges = {}

    for anchor_id in anchor_ids:
        anchor_position = np.asarray(anchors[anchor_id], dtype=float)
        distance_m = np.linalg.norm(anchor_position - position_array)
        distance_m += rng.normal(0.0, noise_std_m)

        ranges[anchor_id] = RangeMeasurement(timestamp_us=1_000_000, 
                                             anchor_id=anchor_id, 
                                             distance_m=float(distance_m),
                                             )

    return RangeSetMeasurement(timestamp_us=1_000_000, 
                               ranges=ranges,
                               )


def test_trilateration_with_eight_anchors(config: TrilaterationConfig) -> None:
    expected_position = (0.4, -0.3, 1.2)
    measurement = create_range_set(config.anchors, expected_position)

    solver = TrilaterationSolver(config)
    result = solver.solve(measurement)

    assert result is not None
    assert result.x == pytest.approx(expected_position[0], abs=1e-4)
    assert result.y == pytest.approx(expected_position[1], abs=1e-4)
    assert result.z == pytest.approx(expected_position[2], abs=1e-4)
    assert result.frame == "FLU"
    assert result.timestamp_us == measurement.timestamp_us



def test_trilateration_with_six_anchors(config: TrilaterationConfig) -> None:
    expected_position = (0.4, -0.3, 1.2)

    # Six anchors distributed across both the lower and upper planes. not correct acording to bitcraze documentation. only dor testing.
    anchor_ids = (0, 1, 2, 4, 5, 6)
    measurement = create_range_set(
        config.anchors,
        expected_position,
        anchor_ids=anchor_ids,
    )

    solver = TrilaterationSolver(config)
    result = solver.solve(measurement)

    assert result is not None
    assert result.x == pytest.approx(expected_position[0], abs=1e-4)
    assert result.y == pytest.approx(expected_position[1], abs=1e-4)
    assert result.z == pytest.approx(expected_position[2], abs=1e-4)


def test_trilateration_rejects_insufficient_anchors(config: TrilaterationConfig) -> None:
    measurement = create_range_set(config.anchors, 
                                   position=(0.4, -0.3, 1.2), 
                                   anchor_ids=(0, 1, 2, 4, 5),
                                   )

    solver = TrilaterationSolver(config)
    result = solver.solve(measurement)

    assert result is None


def test_trilateration_with_noisy_measurements(config: TrilaterationConfig) -> None:
    expected_position = (0.4, -0.3, 1.2)
    measurement = create_range_set(config.anchors, 
                                   expected_position, 
                                   noise_std_m=0.05,
                                   )

    solver = TrilaterationSolver(config)
    result = solver.solve(measurement)

    assert result is not None
    assert result.x == pytest.approx(expected_position[0], abs=0.06)
    assert result.y == pytest.approx(expected_position[1], abs=0.06)
    assert result.z == pytest.approx(expected_position[2], abs=0.06)
