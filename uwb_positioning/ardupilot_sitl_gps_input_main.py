import time
import argparse

from uwb_positioning.config import load_ardupilot_sitl_config

from uwb_positioning.providers.ardupilot_sitl.provider import ArduPilotSitlProvider
from uwb_positioning.providers.ardupilot_sitl.uwb_measurement_degradation import MeasurementDegradation

from uwb_positioning.processing.coordinate_transform import GeographicToNedTransformer, NedToGeographicTransformer
from uwb_positioning.processing.validation import MeasurementValidator

from uwb_positioning.outputs.mavlink_connection import MavlinkConnection
from uwb_positioning.outputs.gps_input import GpsInputOutput

from uwb_positioning.csv_logging.ardupilot_ekf_state import SitlEkfLogger

from analysis.statistics import get_measurement_age_ms, DelayStatistics

from visualization.position_plotter import PositionPlotter

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ArduPilot SITL external position",
                                     description="Run ArduPilot SITL positioning to provide external positioning with the mavlink GPS_INPUT message",
                                     )
    parser.add_argument("--plot-position", action="store_true",
                        help="Plot measured position when program stops",
                        )
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    config = load_ardupilot_sitl_config()

    provider = ArduPilotSitlProvider(config.provider)
    measurement_degradation = MeasurementDegradation(config.simulation)

    geographic_to_ned = GeographicToNedTransformer(config.geographic_transformer)
    ned_to_geographic = NedToGeographicTransformer(config.geographic_transformer)

    validator = MeasurementValidator(config.validator)

    mavlink_connection = MavlinkConnection(config.mavlink)
    odometry_output = GpsInputOutput(mavlink_connection)

    ekf_logger = SitlEkfLogger(config.ekf_logger)

    delay_statistics = DelayStatistics()

    plotter = PositionPlotter() if args.plot_position else None

    send_period_s = 1.0 / config.mavlink.send_rate_hz

    try:
        provider.connect()
        mavlink_connection.connect()

        if not mavlink_connection.wait_for_heartbeat():
            return

        ekf_logger.connect()

        simulation_start = time.monotonic() # start timer for whole simulation
        next_send_time = time.monotonic()
        
        while (config.simulation.duration_s is None
               or time.monotonic() - simulation_start < config.simulation.duration_s
               ):

            raw_measurement = provider.get_measurement()

            if raw_measurement is not None:
                transformed_ned_measurement = geographic_to_ned.transform(raw_measurement)

                noisy_measurement = measurement_degradation.add_gaussian_noise(transformed_ned_measurement)

                measurement_degradation.buffer_measurement(noisy_measurement)

            current_time = time.monotonic()

            # send one message per configured send period
            if current_time >= next_send_time:
                delayed_measurement = measurement_degradation.get_buffer_measurement()

                if delayed_measurement is not None:

                    validated_measurement = validator.validate(delayed_measurement)

                    if validated_measurement.valid:
                        measurement_age_ms = get_measurement_age_ms(validated_measurement)
                        delay_statistics.add_sample(measurement_age_ms)

                        transformed_geographic_measurement = ned_to_geographic.transform(validated_measurement)
                        odometry_output.send(transformed_geographic_measurement)
                    else:
                        print("Measurement rejected: ",
                              f"{validated_measurement.rejection_reason}",
                              )
                    
                    if plotter is not None:
                        plotter.add_measurement(validated_measurement)

                next_send_time += send_period_s

                # catch up loop if send slots get missed
                while next_send_time <= current_time:
                    next_send_time += send_period_s

            ekf_logger.log()

            time.sleep(0.001) # fast polling without overloading CPU

    except KeyboardInterrupt:
        print("\nStopping program...")

    finally:
        ekf_logger.disconnect()
        provider.disconnect()
        mavlink_connection.close()

        print(f"Measurement age:",
              f"Average: {delay_statistics.average_delay_ms:.1f} ms ",
              f"Minimum: {delay_statistics.minimum_delay_ms:.1f} ms ",
              f"Maximum: {delay_statistics.maximum_delay_ms:.1f} ms ",
              )

        if plotter is not None:
            plotter.show_multiview()

if __name__ == "__main__":
    main()
