import time

from config import load_config
from providers.simulator import SimulatedProvider

def main() -> None:
    config = load_config()

    provider = SimulatedProvider(config.simulator)

    loop_frequency_hz = 10.0
    loop_period = 1.0 / loop_frequency_hz

    try:
        provider.connect()
        
        while True:
            loop_start = time.monotonic()

            raw_measurement = provider.get_measurement()
            print(raw_measurement) # Added for testing

            elapsed_time = time.monotonic() - loop_start
            sleep_time = max(0.0, loop_period - elapsed_time)

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping program...")

    finally:
        provider.disconnect()

if __name__ == "__main__":
    main()
