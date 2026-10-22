import csv
import os
import time

from uwb_positioning.config import PositionMeasurementLoggerConfig
from uwb_positioning.models import PositionMeasurement


class PositionMeasurementLogger:
    def __init__(self, config: PositionMeasurementLoggerConfig):
        self.config = config
        self.file = None
        self.writer = None

    def open(self) -> None:
        if self.file is not None:
            return

        output_directory = os.path.dirname(self.config.output_path)

        if not os.path.isdir(output_directory):
            raise FileNotFoundError(f"Save directory does not exist: {output_directory}")

        self.file = open(self.config.output_path,
                         "w",
                         newline="",
                         )
        self.writer = csv.writer(self.file)

        self.writer.writerow(["host_time_ns",
                              "timestamp_us",
                              "x",
                              "y",
                              "z",
                              "frame",
                              "std_x",
                              "std_y",
                              "std_z",
                              "valid",
                              "rejection_reason",
                              ])

    def log(self, measurement: PositionMeasurement) -> None:
        if self.file is None or self.writer is None:
            raise RuntimeError("Position measurement CSV logger is not connected")

        host_time_ns = time.monotonic_ns()

        self.writer.writerow([host_time_ns,
                              measurement.timestamp_us,
                              measurement.x,
                              measurement.y,
                              measurement.z,
                              measurement.frame,
                              measurement.std_x,
                              measurement.std_y,
                              measurement.std_z,
                              measurement.valid,
                              measurement.rejection_reason,
                              ])
        if self.config.flush_every_row:
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()

        self.file = None
        self.writer = None
