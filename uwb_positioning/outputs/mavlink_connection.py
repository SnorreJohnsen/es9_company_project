from typing import cast

from pymavlink import mavutil

from uwb_positioning.config import MavlinkConfig

class MavlinkConnection:
    def __init__(self, config: MavlinkConfig):
        self.config = config
        self.connection: mavutil.mavfile | None = None

    def connect(self) -> None:
        if not self.config.enabled:
            print("MAVLink output disabled")
            return

        print(f"Opening MAVLink connnection:",
              f"{self.config.connection_string}",
              )

        self.connection = cast(mavutil.mavfile, # use cast to tell pyright what to expect since mavfile can return multiple types
                mavutil.mavlink_connection(device=self.config.connection_string, 
                                           source_system=self.config.source_system, 
                                           source_component=self.config.source_component,
                                           )
                               )

    def wait_for_heartbeat(self,
                           timeout: float = 10.0,
                           ) -> bool:
        if not self.config.enabled:
            return True

        if self.connection is None:
            raise RuntimeError("MAVLink connection not opened")

        print("Waiting for ArduPilot heartbeat...")

        heartbeat = self.connection.wait_heartbeat(timeout=timeout)

        if heartbeat is None:
            print("No ArduPilot heartbeat received")
            return False        

        print("Heartbeat received",
              f"{self.connection.target_system=}",
              f"{self.connection.target_component=}",
              )
        return True

    def close(self) -> None:
        connection = self.connection

        if connection is None:
            return

        self.connection = None
        connection.close()
