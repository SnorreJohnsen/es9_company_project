from dataclasses import dataclass


@dataclass
class GeographicMeasurement:
    timestamp_us: int

    latitude_deg: float
    longtitude_deg: float
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
