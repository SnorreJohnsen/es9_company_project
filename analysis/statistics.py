import time

from uwb_positioning.models import PositionMeasurement


def get_measurement_age_ms(measurement: PositionMeasurement) -> float:
    current_time_us = time.monotonic_ns() // 1000
    age_us = current_time_us - measurement.timestamp_us

    return age_us / 1000.0

class DelayStatistics:
    def __init__(self) -> None:
        self.sample_count = 0
        self.total_delay_ms = 0.0
        self.minimum_delay_ms = float("inf")
        self.maximum_delay_ms = 0.0

    def add_sample(self,
                   delay_ms: float,
                   ) -> None:
        self.sample_count += 1
        self.total_delay_ms += delay_ms

        self.minimum_delay_ms = min(self.minimum_delay_ms,
                                    delay_ms,
                                    )

        self.maximum_delay_ms = max(self.maximum_delay_ms,
                                    delay_ms,
                                    )

    @property
    def average_delay_ms(self) -> float:
        if self.sample_count == 0:
            return 0.0

        return self.total_delay_ms / self.sample_count
