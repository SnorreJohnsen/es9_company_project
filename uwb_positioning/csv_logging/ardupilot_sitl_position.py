import time
import csv
import os

from typing import cast
from pymavlink import mavutil

from uwb_positioning.config import SitlLoggerConfig


class SitlLogger:
    def __init__(self, config: SitlLoggerConfig):
        self.config = config

        self.connection: mavutil.mavfile | None = None

        self.ekf_file = None
        self.ekf_writer = None

        self.simstate_file = None
        self.simstate_writer = None

    def connect(self) -> None:
        if self.connection is not None:
            return

        print("Connecting CSV logger to ArduPilot SITL:",
              self.config.connection_string,
              )

        connection = mavutil.mavlink_connection(device=self.config.connection_string)

        self.connection = cast(mavutil.mavfile, connection)

        self.connection.wait_heartbeat()

        print("Logger connected to ArduPilot SITL")

        self._open_ekf_file()
        self._open_simstate_file()

        self._set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                               self.config.message_rate_hz,
                               )
        self._set_message_rate(mavutil.mavlink.MAVLINK_MSG_ID_SIMSTATE,
                               self.config.message_rate_hz,
                               )

    def log(self) -> None:
        connection = self.connection

        if connection is None:
            raise RuntimeError("ArduPilot SITL CSV logger is not connected")


        while True:
            message = connection.recv_match(type=["LOCAL_POSITION_NED",
                                                  "SIMSTATE",
                                                  ],
                                                  blocking=False,
                                                  )
            if message is None:
                break

            host_time_ns = time.monotonic_ns()
            message_type = message.get_type()

            if message_type == "LOCAL_POSITION_NED":
                self._log_ekf_state(message=message,
                                    host_time_ns=host_time_ns,
                                    )
            elif message_type == "SIMSTATE":
                self._log_simstate_state(message=message,
                                         host_time_ns=host_time_ns,
                                         )
            


    def disconnect(self) -> None:
        if self.ekf_file is not None:
            self.ekf_file.close()

        if self.connection is not None:
            self.connection.close()

        self.ekf_file = None
        self.ekf_writer = None
        self.connection = None

    def _open_ekf_file(self) -> None:
        output_directory = os.path.dirname(self.config.ekf_output_path)

        if not os.path.isdir(output_directory):
            raise FileNotFoundError(f"Save directory does not exist: {output_directory}")

        self.ekf_file = open(self.config.ekf_output_path,
                             "w",
                             newline="",
                             )

        self.ekf_writer = csv.writer(self.ekf_file)

        self.ekf_writer.writerow(["host_time_ns",
                                  "sitl_time_ms",
                                  "x_north_m",
                                  "y_east_m",
                                  "z_down_m",
                                  "vx_north_mps",
                                  "vy_east_mps",
                                  "vz_down_mps",
                                  ])

    def _open_simstate_file(self) -> None:
        output_directory = os.path.dirname(self.config.simstate_output_path)

        if not os.path.isdir(output_directory):
            raise FileNotFoundError(f"Save directory does not exist: {output_directory}")

        self.simstate_file = open(self.config.simstate_output_path,
                             "w",
                             newline="",
                             )

        self.simstate_writer = csv.writer(self.simstate_file)

        self.simstate_writer.writerow(["host_time_ns",
                                       "latitude_deg",
                                       "longitude_deg",
                                       "altitude_m",
                                       ])
                                  
    def _set_message_rate(self,
                          message_id: int,
                          rate_hz: float,
                          ) -> None:
        connection = self.connection

        if connection is None:
            raise RuntimeError("SITL positiong logger is not connected")

        interval_us = int(1_000_000 / rate_hz)

        connection.mav.command_long_send(connection.target_system,
                                         connection.target_component,
                                         mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                         0,
                                         message_id,
                                         interval_us,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         )
    def _log_ekf_state(self,
                       message,
                       host_time_ns: int,
                       ) -> None:
        if self.ekf_file is None or self.ekf_writer is None:
            raise RuntimeError("EKF CSV file is not open")

        self.ekf_writer.writerow([host_time_ns,
                                 message.time_boot_ms,
                                 message.x,
                                 message.y,
                                 message.z,
                                 message.vx,
                                 message.vy,
                                 message.vz,
                                 ])
        if self.config.flush_every_row:
            self.ekf_file.flush()

    def _log_simstate_state(self,
                            message,
                            host_time_ns: int,
                            ) -> None:
        if self.simstate_file is None or self.simstate_writer is None:
            raise RuntimeError("SIMSTATE CSV file is not open")

        self.simstate_writer.writerow([host_time_ns,
                                       message.lat,
                                       message.lon,
                                       message.alt,
                                       ])
        if self.config.flush_every_row:
            self.simstate_file.flush()
