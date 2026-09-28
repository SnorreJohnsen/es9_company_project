import matplotlib.pyplot as plt

from models import PositionMeasurement


class PositionPlotter:
    def __init__(self) -> None:
        self.x_values: list[float] = []
        self.y_values: list[float] = []
        self.z_values: list[float] = []

    def add_measurement(self, 
                        measurement: PositionMeasurement,
                        ) -> None:
        self.x_values.append(measurement.x)
        self.y_values.append(measurement.y)
        self.z_values.append(measurement.z)

    def show(self) -> None:
        plt.plot(
                self.x_values,
                self.y_values,
                marker=".",
                )

        plt.xlabel("X position [m]")
        plt.ylabel("Y position [m]")
        plt.title("Simulated position")
