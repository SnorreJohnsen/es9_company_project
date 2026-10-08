import math

from pymavlink import mavutil

from uwb_positioning.models import PositionMeasurement
from uwb_positioning.outputs.mavlink_connection import MavlinkConnection

class OdometryOutput:
    def __init__(self, mavlink_connection: MavlinkConnection):
        self.mavlink_connection = mavlink_connection
        self.reset_counter = 0

    def send(self,
             measurement: PositionMeasurement,
             ) -> None:
        if not measurement.valid:
            return

        config = self.mavlink_connection.config

        if not config.enabled:
            self._print_measurement(measurement)
            return

        connection = self.mavlink_connection.connection

        if connection is None:
            raise RuntimeError("MAVLink connection not available")

        position_covariance = self._create_position_covariance(measurement)

        velocity_covariance = [math.nan] * 21

        connection.mav.odometry_send(time_usec=measurement.timestamp_us,
                                     frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
                                     child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
                                     x=measurement.x,
                                     y=measurement.y,
                                     z=measurement.z,
                                     q=self._unknown_attitude_quaternion(),
                                     vx=math.nan,
                                     vy=math.nan,
                                     vz=math.nan,
                                     rollspeed=math.nan,
                                     pitchspeed=math.nan,
                                     yawspeed=math.nan,
                                     pose_covariance=position_covariance,
                                     velocity_covariance=velocity_covariance,
                                     reset_counter=self.reset_counter,
                                     estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_UNKNOWN,
                                     quality=0, # missing implementation
                                     )

    def notify_estimator_reset(self) -> None:
        self.reset_counter = (self.reset_counter + 1) % 256

    @staticmethod
    def _unknown_attitude_quaternion() -> list:
        return [1, 0, 0, 0] # identity quaternion from ardupilot doc

    @staticmethod
    def _create_position_covariance(measurement: PositionMeasurement,
                                    ) -> list[float]:
        covariance = [0.0] * 21 # create as 0.0 and not math.nan since it causes errors

        if measurement.std_x is not None:
            covariance[0] = measurement.std_x**2

        if measurement.std_y is not None:
            covariance[6] = measurement.std_y**2

        if measurement.std_z is not None:
            covariance[11] = measurement.std_z**2

        # provide large attitude uncertainty because orientation not measured
        covariance[15] = 5.0
        covariance[18] = 5.0
        covariance[20] = 5.0

        return covariance

    @staticmethod
    def _print_measurement(measurement: PositionMeasurement,
                           ) -> None:
        print("ODOMETRY preview | ",
              f"x={measurement.x:7.3f} m | ",
              f"y={measurement.y:7.3f} m | ",
              f"z={measurement.z:7.3f} m | ",
              )
