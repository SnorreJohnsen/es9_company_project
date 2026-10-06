import time

from uwb_positioning.config import load_config
from uwb_positioning.providers.simulator import SimulatedProvider
from uwb_positioning.processing.coordinate_transform import CoordinateTransformer
from uwb_positioning.processing.validation import MeasurementValidator
from uwb_positioning.outputs.mavlink_connection import MavlinkConnection
from uwb_positioning.outputs.odometry import OdometryOutput
from visualization.position_plotter import PositionPlotter

def main() -> None:
    config = load_config()

    provider = SimulatedProvider(config.simulator)

    transformer = CoordinateTransformer(config.transform)

    validator = MeasurementValidator(config.validator)

    mavlink_connection = MavlinkConnection(config.mavlink)

    odometry_output = OdometryOutput(mavlink_connection)

    plotter = PositionPlotter()

    loop_period = 1.0 / config.mavlink.send_rate_hz

    try:
        provider.connect()
        mavlink_connection.connect()

        if not mavlink_connection.wait_for_heartbeat():
            return

        simulation_start = time.monotonic() # start timer for whole simulation
        
        while (config.simulator.duration_s is None
               or time.monotonic() - simulation_start < config.simulator.duration_s
               ):

            loop_start = time.monotonic() # start timer for individual loop

            raw_measurement = provider.get_measurement()

            transformed_measurement = transformer.transform(raw_measurement)

            validated_measurement = validator.validate(transformed_measurement)

            if validated_measurement.valid:
                odometry_output.send(validated_measurement)
            else:
                print("Measurement rejected: ",
                      f"{validated_measurement.rejection_reason}",
                      )
            
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
        plotter.show_multiview()

if __name__ == "__main__":
    main()
