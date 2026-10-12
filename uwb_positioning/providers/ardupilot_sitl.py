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

        message = connection.recv_match(type="SIMSTATE",
                                        blocking=True,
                                        timeout=self.config.message_timeout_s,
                                        )
        if message is None:
            raise TimeoutError('No SIMSTATE message received from ArduPilot SITL')
        
        # drain buffered SIMSTATE messages and use latest one
        while True:
            newest_message = connection.recv_match(type="SIMSTATE",
                                                   blocking=False,
                                                   )

            if newest_message is None:
                break

            message = newest_message

        # prefer integer coordinates for higher precision, fallback to float
        latitude_int = int(getattr(message, "lat_int", 0) or 0)
        longtitude_int = int(getattr(message, "lon_int", 0) or 0)

        if latitude_int != 0:
            latitude_deg = latitude_int / 1e7
        else:
            latitude_deg = float(message.lat)

        if longtitude_int != 0:
            longtitude_deg = longtitude_int / 1e7
        else:
            longtitude_deg = float(message.lon)

        altitude_m = float(message.alt)

        values = (latitude_deg,
                  longtitude_deg,
                  altitude_m,
                  )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("SIMSTATE contains invalid position coordinates")

        if not -90.0 <= latitude_int <= 90.0:
            raise ValueError(f"Invalid latitude: {latitude_deg}")

        if not -180.0 <= longtitude_int <= 180.0:
            raise ValueError(f"Invalid longtitude: {longtitude_deg}")

        # read uncertainty fields if provided, fallback default from config
        horizontal_std_m = float(getattr(message, "std_dev_horz", 0.0) or 0.0)
        vertical_std_m = float(getattr(message, "std_dev_vert", 0.0) or 0.0)

        if (not math.isfinite(horizontal_std_m)
            or horizontal_std_m <= 0.0
            ):
            horizontal_std_m = (self.config.default_horizontal_std_m)

        if (not math.isfinite(vertical_std_m)
            or vertical_std_m <= 0.0
            ):
            vertical_std_m = (self.config.default_vertical_std_m)

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


