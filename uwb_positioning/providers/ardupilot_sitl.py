import math
import time

from typing import cast

from pymavlink import mavutil

from uwb_positioning.config import SitlConfig
from uwb_positioning.models import GeographicMeasurement
from uwb_positioning.providers.base import PositionProvider


class ArduPilotSitlProvider(PositionProvider):
    def __init__(self, config: SitlConfig):
        self.config = config
        self.connection: mavutil.mavfile | None = None

    def connect(self) -> None:
        if self.connection is not None:
            return

        print("Connecting to ArduPilot SITL:",
              self.config.connection_string,
              )

        connection = mavutil.mavlink_connection(device=self.config.connection_string)

        self.connection = cast(mavutil.mavfile, connection) # use cast to tell pyright what to expect since mavfile can return multiple types

        print("ArduPilot SITL connected")

    def get_measurement(self) -> GeographicMeasurement:
        connection = self.connection

        if connection is None:
            raise RuntimeError("ArduPilot SITL is not connected")

        # the dialect does not support altitude 
        # therefore the altitude is obtained through raw simulated gps
        gps_raw_message = connection.recv_match(type="GPS_RAW_INT",
                                                 blocking=True,
                                                 timeout=self.config.message_timeout_s,
                                                 )
        if gps_raw_message is None:
            raise TimeoutError("No GPS_RAW_INT message received from ArduPilot SITL")

        # drain buffered GPS_RAW_INT messages and use latest one
        while True:
            newest_gps_raw_message = connection.recv_match(type="GPS_RAW_INT",
                                                   blocking=False,
                                                   )

            if newest_gps_raw_message is None:
                break

            gps_raw_message = newest_gps_raw_message
        
        latitude_deg = float(gps_raw_message.lat) / 1e7
        longtitude_deg = float(gps_raw_message.lon) / 1e7
        altitude_m = float(gps_raw_message.alt) / 1000.0 # alt is in millimeters

        values = (latitude_deg,
                  longtitude_deg,
                  altitude_m,
                  )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("SITL contains invalid position coordinates")

        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"Invalid latitude: {latitude_deg}")

        if not -180.0 <= longtitude_deg <= 180.0:
            raise ValueError(f"Invalid longtitude: {longtitude_deg}")

        horizontal_std_m = float(gps_raw_message.v_acc) / 1000.0 # GPS_RAW_INT uncertainties are in millimeters
        vertical_std_m = float(gps_raw_message.h_acc) / 1000.0

        return GeographicMeasurement(timestamp_us=time.monotonic_ns() // 1000,
                                     latitude_deg=latitude_deg,
                                     longtitude_deg=longtitude_deg,
                                     altitude_m=altitude_m,
                                     std_horizontal_m=horizontal_std_m,
                                     std_vertical_m=vertical_std_m,
                                     )
    def disconnect(self) -> None:
        connection = self.connection

        if connection is None:
            return

        self.connection = None
        connection.close()


