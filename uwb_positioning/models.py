from dataclasses import dataclass


@dataclass
class GeographicMeasurement:
    timestamp_us: int

    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    std_horizontal_m: float | None = None
    std_vertical_m: float | None = None

@dataclass
class PositionMeasurement:
    timestamp_us: int

    x: float
    y: float
    z: float

    frame: str

    std_x: float | None = None
    std_y: float | None = None
    std_z: float | None = None

    valid: bool = True
    rejection_reason: str | None = None

@dataclass
class RangeMeasurement:
    timestamp_us: int

    anchor_id: int
    distance_m: float

    std_m: float | None = None

@dataclass
class RangeSetMeasurement:
    timestamp_us: int # should represent the mid of the group time range.
    ranges: dict[int, RangeMeasurement]

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("RangeSetMeasuremnt must contaion at least one range")
        for anchor_id, measurement in self.ranges.items():
            if anchor_id != measurement.anchor_id:
                raise ValueError(f"Range key {anchor_id} do not match measurement anchor_id ",
                                 f"{measurement.anchor_id}"
                                 )
