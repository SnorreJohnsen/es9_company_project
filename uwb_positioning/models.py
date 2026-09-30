from dataclasses import dataclass


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
