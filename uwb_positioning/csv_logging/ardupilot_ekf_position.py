import time
import csv
import os

from typing import cast
from pymavlink import mavutil

from uwb_positioning.config import SitlEkfCsvLoggerConfig


class SitlEkfCsvLogger:
    def __init__(self, config: SitlEkfCsvLoggerConfig):
        self.config = config

        self.connection: mavutil.mavfile | None = None
        self.csv_file = None
        self.csv_writer = None

    def connect(self) -> None:
        if self.connection is not None:
            return

        connection = mavutil.mavlink_connection(device=self.config.connection_string)

        self.connection = cast(mavutil.mavfile, connection)

        output_directory = os.path.dirname(self.config.output_path)

        if not os.path.isdir(output_directory):
            raise FileNotFoundError(f"Save directory does not exist: {output_directory}")

        self.csv_file = open(self.config.output_path,
                             "w",
                             newline="",
                             )

        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow(["host_time_ns",
                                  "sitl_time_ms",
                                  "x_north_m",
                                  "y_east_m",
                                  "z_down_m",
                                  "vx_north_mps",
                                  "vy_east_mps",
                                  "vz_down_mps",
                                  ])

    def log(self) -> None:
        if self.connection is None:
            raise RuntimeError("ArduPilot SITL CSV logger is not connected")

        if self.csv_writer is None:
            raise RuntimeError("CSV file is not open")

        while True:
            message = self.connection.recv_match(type="LOCAL_POSITION_NED",
                                                 blocking=False,
                                                 )
            if message is None:
                break

            host_time_ns = time.monotonic_ns()

            self.csv_writer.writerow([host_time_ns,
                                      message.time_boot_ms,
                                      message.x,
                                      message.y,
                                      message.z,
                                      message.vx,
                                      message.vy,
                                      message.vz,
                                      ])
            if self.config.flush_every_row and self.csv_file is not None:
                self.csv_file.flush()

    def disconnect(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()

        if self.connection is not None:
            self.connection.close()

        self.csv_file = None
        self.csv_writer = None
        self.connection = None
                                  
