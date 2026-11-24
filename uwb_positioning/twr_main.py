import time

from uwb_positioning.config import load_simulator_config

#from uwb_positioning.providers.bitcraze_loco_twr.provider import BitcrazeLocoTwrProvider
from uwb_positioning.providers.simulator.provider import SimulatedProvider

from uwb_positioning.processing.coordinate_transform import NedToGeographicTransformer
from uwb_positioning.processing.validation import MeasurementValidator

from uwb_positioning.outputs.mavlink_connection import MavlinkConnection
from uwb_positioning.outputs.gps_input import GpsInputOutput


def main() -> None:

    config = load_simulator_config()

    provider = SimulatedProvider(config.provider)

    ned_to_geographic = NedToGeographicTransformer(config.geographic_transformer)

    validator = MeasurementValidator(config.validator)

    mavlink_connection = MavlinkConnection(config.mavlink)
    gps_output = GpsInputOutput(mavlink_connection)

    send_period_s = 1.0 / config.mavlink.send_rate_hz

    try:
        provider.connect()
        mavlink_connection.connect()

        if not mavlink_connection.wait_for_heartbeat():
            return

        simulation_start = time.monotonic() # start timer for whole simulation
        next_send_time = time.monotonic()
        
        while (config.provider.duration_s is None
               or time.monotonic() - simulation_start < config.provider.duration_s
               ):

            raw_measurement = provider.get_measurement()

            current_time = time.monotonic()

            # send one message per configured send period
            if current_time >= next_send_time:

                validated_measurement = validator.validate(raw_measurement)

                if validated_measurement.valid:
                    transformed_geographic_measurement = ned_to_geographic.transform(validated_measurement)
                    gps_output.send(transformed_geographic_measurement)

                else:
                    print("Measurement rejected: ",
                          f"{validated_measurement.rejection_reason}",
                          )
                
                next_send_time += send_period_s

                # catch up loop if send slots get missed
                while next_send_time <= current_time:
                    next_send_time += send_period_s

            time.sleep(0.001) # fast polling without overloading CPU

    except KeyboardInterrupt:
        print("\nStopping program...")

    finally:
        provider.disconnect()
        mavlink_connection.close()

if __name__ == "__main__":
    main()
