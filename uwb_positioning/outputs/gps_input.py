from pymavlink import mavutil

from uwb_positioning.models import GeographicMeasurement
from uwb_positioning.outputs.mavlink_connection import MavlinkConnection


class GpsInputOutput:
    def __init__(self, mavlink_connection: MavlinkConnection):
        self.mavlink_connection = mavlink_connection

    def send(self, 
             measurement: GeographicMeasurement,
             ) -> None:
        config = self.mavlink_connection.config

        if not config.enabled:
            self._print_measurement(measurement)
            return

        connection = self.mavlink_connection.connection

        if connection is None:
            raise RuntimeError("MAVLink connection not available")

        ignore_flags = self._create_ignore_flags(measurement)

        connection.mav.gps_input_send(time_usec=measurement.timestamp_us,
                                      gps_id=0,
                                      ignore_flags=ignore_flags,
                                      time_week_ms=0,
                                      time_week=0,
                                      fix_type=3,
                                      lat=round(measurement.latitude_deg * 1e7),
                                      lon=round(measurement.longitude_deg * 1e7),
                                      alt=measurement.altitude_m,
                                      hdop=1.21, # needs for gps health check,, taken form gps2_raw.
                                      vdop=2.00, # should be changed for uwb to represent acutal uncertainty
                                      vn=0.0,
                                      ve=0.0,
                                      vd=0.0,
                                      speed_accuracy=0.0,
                                      horiz_accuracy=(measurement.std_horizontal_m
                                                      if measurement.std_horizontal_m is not None
                                                      else 0.0
                                                      ),
                                      vert_accuracy=(measurement.std_vertical_m 
                                                     if measurement.std_vertical_m is not None 
                                                     else 0.0
                                                     ),
                                      satellites_visible=10, #should maybe be changed to anchors.
                                      yaw=0, # should be implemented if yaw is estimated
                                      )


    @staticmethod
    def _create_ignore_flags(measurement: GeographicMeasurement) -> int:
        ignore_flags = (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ
                        | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT
                        | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY
                        )

        if measurement.std_horizontal_m is None:
            ignore_flags |= (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY)

        if measurement.std_vertical_m is None:
            ignore_flags |= (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)

        return ignore_flags

    @staticmethod
    def _print_measurement(measurement: GeographicMeasurement) -> None:
        print("GPS_INPUT preview | ",
              f"lat={measurement.latitude_deg:11.7f} deg | ",
              f"lon={measurement.longitude_deg:11.7f} deg | ",
              f"alt={measurement.altitude_m:11.7f} m | ",
              )
