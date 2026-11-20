import numpy as np
from scipy.optimize import least_squares

from uwb_positioning.config import TrilaterationConfig
from uwb_positioning.models import RangeSetMeasurement, PositionMeasurement


class TrilaterationSolver:
    def __init__(self, config: TrilaterationConfig) -> None:
        self.config = config
        self._last_position: np.ndarray | None = None

        self._anchors = {anchor_id: np.asarray(position, dtype=float)
                         for anchor_id, position in config.anchors.items()
                         }

    @staticmethod
    def _residuals(position: np.ndarray,
                   anchors: np.ndarray,
                   distances: np.ndarray,
                   ) -> np.ndarray:

        calculated_distances = np.linalg.norm(anchors - position,
                                              axis=1,
                                              )
        return calculated_distances - distances

    def solve(self,
              measurement: RangeSetMeasurement,
              ) -> PositionMeasurement | None:
        
        # collect valid measurements
        samples = [(self._anchors[anchor_id], range_measurement.distance_m)
                   for anchor_id, range_measurement in measurement.ranges.items()
                   if anchor_id in self._anchors
                   and np.isfinite(range_measurement.distance_m)
                   and range_measurement.distance_m > 0
                   ]

        if len(samples) < self.config.minimum_anchors:
            return None

        anchors = np.asarray([sample[0] for sample in samples])
        distances = np.asarray([sample[1] for sample in samples])

        initial_position = (self._last_position.copy()
                            if self._last_position is not None
                            else np.mean(anchors, axis=0)
                            )

        result = least_squares(self._residuals,
                               x0=initial_position,
                               args=(anchors, distances),
                               method="trf",
                               loss=self.config.loss,
                               f_scale=self.config.loss_scale_m,
                               max_nfev=self.config.max_nfev,
                               )

        if not result.success:
            return None

        residual_rms = np.sqrt(np.mean(result.fun**2))

        # reject optionally max residual error
        if (self.config.max_residual_rms_m is not None 
            and residual_rms > self.config.max_residual_rms_m
            ):
            return None

        position = result.x
        self._last_position = position.copy()

        return PositionMeasurement(timestamp_us=measurement.timestamp_us,
                                   x=float(position[0]),
                                   y=float(position[1]),
                                   z=float(position[2]),
                                   frame="FLU",
                                   )
