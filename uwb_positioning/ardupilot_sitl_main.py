import time
import argparse

from uwb_positioning.config import load_ardupilot_sitl_config

from uwb_positioning.providers.ardupilot_sitl.provider import ArduPilotSitlProvider
from uwb_positioning.providers.ardupilot_sitl.uwb_measurement_degradation import MeasurementDegradation

from uwb_positioning.processing.coordinate_transform import GeographicToNedTransformer
from uwb_positioning.processing.validation import MeasurementValidator

from uwb_positioning.outputs.mavlink_connection import MavlinkConnection
from uwb_positioning.outputs.odometry import OdometryOutput

from analysis.statistics import get_measurement_age_ms, DelayStatistics

from visualization.position_plotter import PositionPlotter

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ArdyPilot SITL external position",
                                     description="Run ArduPilot SITL positioning to provide external positioning with the mavlink odometry message",
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

    transformer = GeographicToNedTransformer()

    validator = MeasurementValidator(config.validator)

    mavlink_connection = MavlinkConnection(config.mavlink)
    odometry_output = OdometryOutput(mavlink_connection)

    delay_statistics = DelayStatistics()

    plotter = PositionPlotter() if args.plot_position else None

    loop_period = 1.0 / config.mavlink.send_rate_hz

    try:
        provider.connect()
        mavlink_connection.connect()

        if not mavlink_connection.wait_for_heartbeat():
            return

        simulation_start = time.monotonic() # start timer for whole simulation
        
        while (config.simulation.duration_s is None
               or time.monotonic() - simulation_start < config.simulation.duration_s
               ):

            loop_start = time.monotonic() # start timer for individual loop

            raw_measurement = provider.get_measurement()

            transformed_measurement = transformer.transform(raw_measurement)

            noisy_measurement = measurement_degradation.add_gaussian_noise(transformed_measurement)

            delayed_measurement = measurement_degradation.add_fixed_delay(noisy_measurement)

            if delayed_measurement is not None:

                validated_measurement = validator.validate(delayed_measurement)

                if validated_measurement.valid:
                    measurement_age_ms = get_measurement_age_ms(validated_measurement)
                    delay_statistics.add_sample(measurement_age_ms)

                    odometry_output.send(validated_measurement)
                else:
                    print("Measurement rejected: ",
                          f"{validated_measurement.rejection_reason}",
                          )
                
                if plotter is not None:
                    plotter.add_measurement(validated_measurement)

            # sleep time to ensure loop runs at the configured frequency
            elapsed_time = time.monotonic() - loop_start
            sleep_time = max(0.0, loop_period - elapsed_time)

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping program...")

    finally:
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
