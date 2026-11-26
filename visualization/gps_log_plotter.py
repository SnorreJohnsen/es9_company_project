import argparse
import pandas as pd
import matplotlib.pyplot as plt

from uwb_positioning.config import GeographicTransformConfig
from uwb_positioning.models import GeographicMeasurement, PositionMeasurement
from uwb_positioning.processing.coordinate_transform import GeographicToNedTransformer


def parse_arduments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="GPS log plotter",
                                     description="Plots GPS CSV data from cubepilot log file",
                                     )
    parser.add_argument("filename",
                        help="Path to GPS CSV log file",
                        )
    return parser.parse_args()

def extract_geographic_measurements(df: pd.DataFrame) -> list[GeographicMeasurement]:

    geo_measurements = []

    for _, row in df.iterrows():
        measurement = GeographicMeasurement(
            timestamp_us=int(row["TimeUS"]),
            latitude_deg=float(row["Lat"]),
            longitude_deg=float(row["Lng"]),
            altitude_m=float(row["Alt"]),
        )

        geo_measurements.append(measurement)

    return geo_measurements

def convert_geo_measurements_to_ned(geo_measurements: list[GeographicMeasurement]) -> list[PositionMeasurement]:

    geographic_to_ned = GeographicToNedTransformer(GeographicTransformConfig())

    ned_measurements = []

    for measurement in geo_measurements:
        ned_measurement = geographic_to_ned.transform(measurement)
        ned_measurements.append(ned_measurement)
    
    return ned_measurements

def main() -> None:
    args = parse_arduments()

    df = pd.read_csv(args.filename)

    geo_measurements = extract_geographic_measurements(df)
    ned_measurements = convert_geo_measurements_to_ned(geo_measurements)

    plt.figure(figsize=(12, 7))

    north = [measurement.x for measurement in ned_measurements]
    east = [measurement.y for measurement in ned_measurements]

    plt.plot(east, north, marker=".", markersize=3)

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("CubePilot GPS trajectory (NED)")
    plt.axis("equal")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
