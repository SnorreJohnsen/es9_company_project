import os
import matplotlib.pyplot as plt

from uwb_positioning.models import PositionMeasurement


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

    def show_xy(self,
                save_path: str | None = None,
                ) -> None:
        figure, axes = plt.subplots()

        self._plot_xy(axes)

        figure.tight_layout()

        self._save_or_show_fig(figure, save_path)

    def show_3d(self,
                save_path: str | None = None,
                ) -> None:
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d")

        self._plot_3d(axes)

        figure.tight_layout()

        self._save_or_show_fig(figure, save_path)
    
    def show_multiview(self,
                       save_path: str | None = None,
                       ) -> None:
        figure = plt.figure(figsize=(12, 7))

        axes_3d = figure.add_subplot(2, 2, 1, projection="3d")
        axes_xy = figure.add_subplot(2, 2, 2)
        axes_xz = figure.add_subplot(2, 2, 3)
        axes_yz = figure.add_subplot(2, 2, 4)

        self._plot_3d(axes_3d)
        self._plot_xy(axes_xy)
        self._plot_xz(axes_xz)
        self._plot_yz(axes_yz)

        figure.suptitle("Simulated position")
        figure.tight_layout()

        self._save_or_show_fig(figure, save_path)

    def _save_or_show_fig(self,
                          figure,
                          save_path: str | None,
                          ) -> None:
        if save_path is None:
            plt.show()
            return

        if not os.path.isdir(os.path.dirname(save_path)):
            raise FileNotFoundError(f"Save directory does not exist: {os.path.dirname(save_path)}")

        figure.savefig(save_path)
        plt.close(figure)

    def _plot_xy(self, axes) -> None:
        axes.plot(self.x_values, 
                  self.y_values, 
                  marker=".",
                  )

        axes.set_xlabel("X position [m]")
        axes.set_ylabel("Y position [m]")
        axes.set_title("XY")
        axes.grid()

    def _plot_xz(self, axes) -> None:
        axes.plot(self.x_values, 
                  self.z_values, 
                  marker=".",
                  )

        axes.set_xlabel("X position [m]")
        axes.set_ylabel("Z position [m]")
        axes.set_title("XZ")
        axes.grid()

    def _plot_yz(self, axes) -> None:
        axes.plot(self.y_values, 
                  self.z_values, 
                  marker=".",
                  )

        axes.set_xlabel("Y position [m]")
        axes.set_ylabel("Z position [m]")
        axes.set_title("YZ")
        axes.grid()

    def _plot_3d(self, axes) -> None:
        axes.plot(self.x_values,
                  self.y_values,
                  self.z_values,
                  marker=".",
                  )

        axes.set_xlabel("X position [m]")
        axes.set_ylabel("Y position [m]")
        axes.set_zlabel("Z position [m]")
        axes.set_title("3D")
