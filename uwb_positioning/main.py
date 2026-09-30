import time

from uwb_positioning.config import load_config
from uwb_positioning.providers.simulator import SimulatedProvider
from uwb_positioning.processing.coordinate_transform import CoordinateTransformer
from visualization.position_plotter import PositionPlotter

def main() -> None:
    config = load_config()

    provider = SimulatedProvider(config.simulator)

    transformer = CoordinateTransformer(config.transform)

    plotter = PositionPlotter()

    loop_frequency_hz = 10.0
    loop_period = 1.0 / loop_frequency_hz

    try:
        provider.connect()
        
        while True:
            loop_start = time.monotonic()

            raw_measurement = provider.get_measurement()

            transformed_measurement = transformer.transform(raw_measurement)
            
            plotter.add_measurement(transformed_measurement)

            elapsed_time = time.monotonic() - loop_start
            sleep_time = max(0.0, loop_period - elapsed_time)

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping program...")

    finally:
        provider.disconnect()
        plotter.show_multiview()

if __name__ == "__main__":
    main()
