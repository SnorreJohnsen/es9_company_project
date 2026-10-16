import math

from uwb_positioning.config import ValidationConfig
from uwb_positioning.models import PositionMeasurement

class MeasurementValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.previous_measurement: PositionMeasurement | None = None

    def validate(self,
                 measurement: PositionMeasurement,
                 ) -> PositionMeasurement:
        rejection_reason = self._find_rejection_reason(measurement)

        measurement.valid = rejection_reason is None
        measurement.rejection_reason = rejection_reason

        if measurement.valid:
            self.previous_measurement = measurement

        return measurement

    def _find_rejection_reason(self,
                               measurement: PositionMeasurement,
                               ) -> str | None:
        if not self._contains_finite_values(measurement):
            return "Measurement containts NaN or infinite values"

        if not self._is_inside_allowed_area(measurement):
            return "Measurement outside flight area"

        if self.previous_measurement is not None:
            timing_error = self._check_timestamp(measurement)

            if timing_error is not None:
                return timing_error

            if self._exceeds_maximum_velocity(measurement):
                return "Velocity exceeds configured limit"

        return None

    def _contains_finite_values(self,
                                measurement: PositionMeasurement,
                                ) -> bool:
        return all(math.isfinite(value)
                   for value in (measurement.x,
                                 measurement.y,
                                 measurement.z
                                 )
                   )

    def _check_timestamp(self,
                         measurement: PositionMeasurement,
                         ) -> str | None:
        previous = self.previous_measurement

        if previous is None:
            return None

        if measurement.timestamp_us <= previous.timestamp_us:
            return "Timestamp is not newer than previous measurement"

        return None

    def _is_inside_allowed_area(self,
                                measurement: PositionMeasurement,
                                ) -> bool:
        return (
            self.config.x_min <= measurement.x <= self.config.x_max
            and self.config.y_min <= measurement.y <= self.config.y_max
            and self.config.z_min <= measurement.z <= self.config.z_max
        )

    def _exceeds_maximum_velocity(self,
                                  measurement: PositionMeasurement,
                                  ) -> bool:
        previous = self.previous_measurement

        if previous is None:
            return False

        delta_time = (measurement.timestamp_us - previous.timestamp_us) / 1_000_000.0

        if delta_time <= 0.0:
            return True

        delta_x = measurement.x - previous.x
        delta_y = measurement.y - previous.y
        delta_z = measurement.z - previous.z

        distance = math.sqrt(delta_x**2
                             + delta_y**2
                             + delta_z**2
                             )

        velocity = distance / delta_time

        return velocity > self.config.maximum_velocity
