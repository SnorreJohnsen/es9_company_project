"""Plot external position measurements, ArduPilot EKF, and SIMSTATE logs."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from uwb_positioning.config import (
    GeographicTransformConfig,
    PositionMeasurementLoggerConfig,
    SitlEkfPlotterConfig,
    SitlLoggerConfig,
)
from uwb_positioning.processing.coordinate_transform import GeographicToNedTransformer


class EKFPlotter:
    def __init__(self,
                 config: SitlEkfPlotterConfig,
                 geographic_config: GeographicTransformConfig,
                 ) -> None:
        self.config = config
        self.geographic_config = geographic_config
        self.geographic_transformer = GeographicToNedTransformer(geographic_config)

        self.position_measurement: pd.DataFrame | None = None
        self.sitl_ekf_state: pd.DataFrame | None = None
        self.sitl_simstate: pd.DataFrame | None = None

    def load_csv_files(self,
                       position_measurement: str | None = None,
                       sitl_ekf_state: str | None = None,
                       sitl_simstate: str | None = None,
                       ) -> "EKFPlotter":
        self.position_measurement = (
            self._prepare_position(self._read_csv(position_measurement))
            if position_measurement else None
        )
        self.sitl_ekf_state = (
            self._prepare_ekf(self._read_csv(sitl_ekf_state))
            if sitl_ekf_state else None
        )
        self.sitl_simstate = (
            self._prepare_simstate(self._read_csv(sitl_simstate))
            if sitl_simstate else None
        )

        return self

    def position_vs_time(self) -> tuple[Figure, np.ndarray]:
        """Plot north, east, and down at each source's native sample rate."""
        self._require_data()

        fig, axes = plt.subplots(
            3, 1,
            figsize=(self.config.figure_width, self.config.figure_height),
            sharex=True,
        )

        position = self._usable_position()
        ekf = self._usable_ekf()

        for axis, component, label in zip(
                axes,
                ("north_m", "east_m", "down_m"),
                ("North (m)", "East (m)", "Down (m)"),
                ):
            if position is not None and not position.empty:
                axis.plot(
                    self._time_seconds(self._series(position, "host_time_ns")),
                    self._series(position, component),
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    label="External position measurement",
                )

            if ekf is not None and not ekf.empty:
                axis.plot(
                    self._time_seconds(self._series(ekf, "host_time_ns")),
                    self._series(ekf, component),
                    label="EKF",
                )

            if component != "down_m" and self.sitl_simstate is not None:
                simstate = self.sitl_simstate
                if not simstate.empty:
                    axis.plot(
                        self._time_seconds(self._series(simstate, "host_time_ns")),
                        self._series(simstate, component),
                        label="SIMSTATE truth",
                    )

            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)

            if axis.lines:
                axis.legend()

        axes[-1].set_xlabel(
            "Time since first external measurement (s)"
            if position is not None and not position.empty
            else "Time since first loaded sample (s)"
        )

        fig.suptitle("Position vs time (native sample rates)")
        fig.tight_layout()
        self._save_figure(fig, "position_vs_time.png")

        return fig, axes

    def errors_vs_time(self) -> tuple[Figure, np.ndarray]:
        """Plot external-measurement and EKF errors against SIMSTATE."""
        aligned = self._align_to_position_timeline()

        fig, axes = plt.subplots(
            3, 1,
            figsize=(self.config.figure_width, self.config.figure_height),
            sharex=True,
        )

        panels = (
            ("north_error_m", "North error (m)"),
            ("east_error_m", "East error (m)"),
            ("horizontal_error_m", "2D horizontal error (m)"),
        )
        time_s = self._time_seconds(self._series(aligned, "host_time_ns"))

        for axis, (error_name, label) in zip(axes, panels):
            sources = (
                ("measurement", "External measurement − SIMSTATE"),
                ("ekf", "EKF − SIMSTATE"),
            )

            for prefix, source_label in sources:
                column = f"{prefix}_{error_name}"
                if column not in aligned.columns:
                    continue

                values = self._numeric(aligned, column)
                finite_values = np.isfinite(
                    values.to_numpy(dtype=float, na_value=np.nan)
                )
                if not finite_values.any():
                    continue

                if prefix == "measurement":
                    axis.plot(
                        time_s,
                        values,
                        label=source_label,
                    )
                else:
                    # Markers keep isolated samples visible across gaps.
                    axis.plot(
                        time_s,
                        values,
                        label=source_label,
                    )

            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)

            if axis.lines:
                axis.legend()

        axes[-1].set_xlabel("Time since first external measurement (s)")
        fig.suptitle("Position and EKF errors relative to SIMSTATE truth")
        fig.tight_layout()
        self._save_figure(fig, "errors_vs_time.png")

        return fig, axes

    def topdown_trajectory(self) -> tuple[Figure, Axes]:
        """Plot all available trajectories at each source's native sample rate."""
        self._require_data()

        fig, axis = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height * 0.8),
        )

        position = self._usable_position()
        if position is not None and not position.empty:
            axis.plot(
                self._series(position, "east_m"),
                self._series(position, "north_m"),
                linestyle="none",
                marker="o",
                markersize=3,
                label="External position measurement",
            )

        ekf = self._usable_ekf()
        if ekf is not None and not ekf.empty:
            axis.plot(
                self._series(ekf, "east_m"),
                self._series(ekf, "north_m"),
                label="EKF",
            )

        if self.sitl_simstate is not None and not self.sitl_simstate.empty:
            axis.plot(
                self._series(self.sitl_simstate, "east_m"),
                self._series(self.sitl_simstate, "north_m"),
                label="SIMSTATE truth",
            )

        if not axis.lines:
            raise RuntimeError("No position data is available for the trajectory plot")

        axis.set_title("Top-down trajectory")
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        axis.legend()

        fig.tight_layout()
        self._save_figure(fig, "topdown_trajectory.png")

        return fig, axis

    def error_distributions(self) -> tuple[Figure, np.ndarray]:
        """Plot error histograms for external measurements and EKF vs SIMSTATE."""
        aligned = self._align_to_position_timeline()

        fig, axes = plt.subplots(
            1, 3,
            figsize=(self.config.figure_width * 1.35,
                     self.config.figure_height * 0.65),
        )

        panels = (
            ("north_error_m", "North error (m)"),
            ("east_error_m", "East error (m)"),
            ("horizontal_error_m", "2D horizontal error (m)"),
        )

        for axis, (error_name, title) in zip(axes, panels):
            histograms: list[tuple[str, np.ndarray]] = []
            sources = (
                ("measurement", "External measurement − SIMSTATE"),
                ("ekf", "EKF − SIMSTATE"),
            )

            for prefix, label in sources:
                column = f"{prefix}_{error_name}"
                if column not in aligned.columns:
                    continue

                values = self._numeric(aligned, column).to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
                values = values[np.isfinite(values)]
                if values.size:
                    histograms.append((label, values))

            if histograms:
                combined = np.concatenate(
                    [values for _, values in histograms]
                )

                if np.min(combined) == np.max(combined):
                    center = float(combined[0])
                    bins = np.linspace(center - 0.05, center + 0.05, 11)
                else:
                    bins = np.histogram_bin_edges(
                        combined,
                        bins=self.config.histogram_bins,
                    )

                for label, values in histograms:
                    axis.hist(
                        values,
                        bins=bins,
                        density=True,
                        alpha=0.5,
                        label=label,
                    )
                axis.legend()
            else:
                axis.text(
                    0.5, 0.5,
                    "No matched samples",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

            axis.set_title(title)
            axis.set_xlabel(title)
            axis.set_ylabel("Probability density")
            axis.grid(True, alpha=0.25)

        fig.suptitle("Position and EKF error distributions relative to SIMSTATE")
        fig.tight_layout()
        self._save_figure(fig, "error_distributions.png")

        return fig, axes

    @staticmethod
    def _read_csv(filename: str) -> pd.DataFrame:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"CSV file does not exist: {filename}")

        frame = pd.read_csv(filename)
        frame.columns = [str(column).strip() for column in frame.columns]

        if frame.empty:
            raise RuntimeError(f"CSV file is empty: {filename}")

        return frame

    @staticmethod
    def _series(frame: pd.DataFrame, column: str) -> pd.Series:
        result = frame[column]
        if not isinstance(result, pd.Series):
            raise RuntimeError(f"Duplicate CSV column name: {column}")
        return result

    @classmethod
    def _numeric(cls,
                 frame: pd.DataFrame,
                 column: str,
                 ) -> pd.Series:
        values = pd.to_numeric(cls._series(frame, column), errors="coerce")
        if not isinstance(values, pd.Series):
            raise RuntimeError(f"Could not read numeric CSV column: {column}")
        return values

    @classmethod
    def _prepare_time(cls,
                      frame: pd.DataFrame,
                      name: str,
                      ) -> pd.DataFrame:
        if "host_time_ns" not in frame.columns:
            raise RuntimeError(f"{name} CSV must contain host_time_ns")

        result = frame.copy()
        result["host_time_ns"] = cls._numeric(result, "host_time_ns")
        result = result.dropna(subset=["host_time_ns"])

        if result.empty:
            raise RuntimeError(f"{name} CSV has no valid host timestamps")

        result["host_time_ns"] = result["host_time_ns"].astype("int64")
        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _prepare_position(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_time(frame, "Position measurement")
        missing = [column for column in ("x", "y", "z")
                   if column not in result.columns]
        if missing:
            raise RuntimeError(
                f"Position CSV missing columns: {', '.join(missing)}"
            )

        if "frame" in result.columns:
            frames = {
                str(value).strip().upper()
                for value in result["frame"].dropna()
            }
            if frames - {"NED"}:
                raise RuntimeError(
                    "Position measurements must be transformed to NED first"
                )

        result["north_m"] = self._numeric(result, "x")
        result["east_m"] = self._numeric(result, "y")
        result["down_m"] = self._numeric(result, "z")

        if "valid" in result.columns:
            result["_valid_bool"] = result["valid"].map(self._parse_bool)
        else:
            result["_valid_bool"] = True

        return result

    def _prepare_ekf(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_time(frame, "SITL EKF")

        if all(column in result.columns for column in (
                "x_north_m", "y_east_m", "z_down_m")):
            north, east, down = "x_north_m", "y_east_m", "z_down_m"
        elif all(column in result.columns for column in ("x", "y", "z")):
            north, east, down = "x", "y", "z"
        else:
            raise RuntimeError(
                "EKF CSV needs x_north_m/y_east_m/z_down_m or x/y/z"
            )

        result["north_m"] = self._numeric(result, north)
        result["east_m"] = self._numeric(result, east)
        result["down_m"] = self._numeric(result, down)

        if "sitl_time_ms" in result.columns:
            result["sitl_time_ms"] = self._numeric(result, "sitl_time_ms")

        return result

    def _prepare_simstate(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_time(frame, "SITL SIMSTATE")
        missing = [column for column in ("latitude_deg", "longitude_deg")
                   if column not in result.columns]
        if missing:
            raise RuntimeError(
                f"SIMSTATE CSV missing columns: {', '.join(missing)}"
            )

        result["latitude_deg"] = self._numeric(result, "latitude_deg")
        result["longitude_deg"] = self._numeric(result, "longitude_deg")
        result = result.dropna(subset=["latitude_deg", "longitude_deg"])

        result["north_m"] = self.geographic_transformer.latitude_to_north(
            self._series(result, "latitude_deg")
        )
        result["east_m"] = self.geographic_transformer.longitude_to_east(
            self._series(result, "longitude_deg")
        )

        return result.reset_index(drop=True)

    @staticmethod
    def _parse_bool(value: object) -> bool:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return False

        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value) and bool(np.isfinite(value))

        return str(value).strip().lower() in {
            "true", "t", "yes", "y", "1", "valid"
        }

    def _usable_position(self) -> pd.DataFrame | None:
        if self.position_measurement is None:
            return None

        result = self.position_measurement
        if not self.config.include_invalid_measurements:
            result = result.loc[self._series(result, "_valid_bool"), :]

        columns = ["north_m", "east_m", "down_m"]
        result = result.dropna(subset=columns)
        if not result.empty:
            finite = np.isfinite(result[columns].astype(float)).all(axis=1)
            result = result.loc[finite, :]

        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _usable_ekf(self) -> pd.DataFrame | None:
        if self.sitl_ekf_state is None:
            return None

        result = self.sitl_ekf_state.dropna(
            subset=["north_m", "east_m", "down_m"]
        )
        if not result.empty:
            columns = ["north_m", "east_m", "down_m"]
            finite = np.isfinite(result[columns].astype(float)).all(axis=1)
            result = result.loc[finite, :]

        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _align_to_position_timeline(self) -> pd.DataFrame:
        """Match EKF and SIMSTATE to each external measurement timestamp."""
        position = self._usable_position()
        if position is None or position.empty:
            raise RuntimeError("Error plots require external position measurements")

        aligned = pd.DataFrame({
            "host_time_ns": self._series(position, "host_time_ns"),
            "measurement_north_m": self._series(position, "north_m"),
            "measurement_east_m": self._series(position, "east_m"),
            "measurement_down_m": self._series(position, "down_m"),
        })

        tolerance = (
            None if self.config.sync_tolerance_ms is None
            else int(self.config.sync_tolerance_ms * 1_000_000)
        )

        ekf = self._usable_ekf()
        if ekf is not None and not ekf.empty:
            ekf_samples = pd.DataFrame({
                "ekf_host_time_ns": self._series(ekf, "host_time_ns"),
                "ekf_north_m": self._series(ekf, "north_m"),
                "ekf_east_m": self._series(ekf, "east_m"),
                "ekf_down_m": self._series(ekf, "down_m"),
            })

            aligned = pd.merge_asof(
                aligned.sort_values("host_time_ns"),
                ekf_samples.sort_values("ekf_host_time_ns"),
                left_on="host_time_ns",
                right_on="ekf_host_time_ns",
                direction="nearest",
                tolerance=tolerance,
            )

        if self.sitl_simstate is None or self.sitl_simstate.empty:
            raise RuntimeError("Error plots require SIMSTATE ground truth")

        truth = self.sitl_simstate
        truth_samples = pd.DataFrame({
            "simstate_host_time_ns": self._series(truth, "host_time_ns"),
            "truth_north_m": self._series(truth, "north_m"),
            "truth_east_m": self._series(truth, "east_m"),
        })

        aligned = pd.merge_asof(
            aligned.sort_values("host_time_ns"),
            truth_samples.sort_values("simstate_host_time_ns"),
            left_on="host_time_ns",
            right_on="simstate_host_time_ns",
            direction="nearest",
            tolerance=tolerance,
        )

        sources = (
            ("measurement", "measurement_north_m", "measurement_east_m"),
            ("ekf", "ekf_north_m", "ekf_east_m"),
        )
        for prefix, north, east in sources:
            if north not in aligned.columns or east not in aligned.columns:
                continue

            aligned[f"{prefix}_north_error_m"] = (
                aligned[north] - aligned["truth_north_m"]
            )
            aligned[f"{prefix}_east_error_m"] = (
                aligned[east] - aligned["truth_east_m"]
            )
            aligned[f"{prefix}_horizontal_error_m"] = np.sqrt(
                aligned[f"{prefix}_north_error_m"] ** 2
                + aligned[f"{prefix}_east_error_m"] ** 2
            )

        return aligned.reset_index(drop=True)

    def print_alignment_diagnostics(self) -> None:
        position = self._usable_position()
        if position is None or position.empty or self.sitl_simstate is None:
            return

        aligned = self._align_to_position_timeline()
        total = len(aligned)
        print(f"Alignment to external measurements: {total} timeline samples.")

        sources = (
            ("EKF", "ekf_host_time_ns", "ekf_north_m"),
            ("SIMSTATE", "simstate_host_time_ns", "truth_north_m"),
        )
        for name, time_column, value_column in sources:
            if time_column not in aligned.columns:
                print(f"  {name}: no samples available.")
                continue

            matched = aligned[time_column].notna() & aligned[value_column].notna()
            count = int(matched.sum())
            print(f"  {name}: matched {count}/{total} samples.")

            if count:
                offsets = (
                    aligned.loc[matched, time_column]
                    - aligned.loc[matched, "host_time_ns"]
                ) / 1e6
                print(
                    f"    offset ms min/median/max: {offsets.min():.2f}/"
                    f"{offsets.median():.2f}/{offsets.max():.2f}"
                )

        print(f"  Tolerance: {self.config.sync_tolerance_ms} ms")

    def _require_data(self) -> None:
        if all(data is None for data in (
                self.position_measurement,
                self.sitl_ekf_state,
                self.sitl_simstate,
                )):
            raise RuntimeError("Load at least one CSV file before plotting")

    def _time_origin_ns(self) -> int:
        position = self._usable_position()
        if position is not None and not position.empty:
            return int(position["host_time_ns"].min())

        starts = [
            int(data["host_time_ns"].min())
            for data in (self.sitl_ekf_state, self.sitl_simstate)
            if data is not None and not data.empty
        ]
        if not starts:
            raise RuntimeError("No usable timestamps in the loaded CSV files")

        return min(starts)

    def _time_seconds(self, timestamps: pd.Series) -> pd.Series:
        return (
            timestamps.astype("int64") - self._time_origin_ns()
        ) / 1_000_000_000.0

    def _save_figure(self,
                     figure: Figure,
                     filename: str,
                     ) -> None:
        directory = self.config.output_directory
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Plot output directory does not exist: {directory}"
            )

        output_file = os.path.join(directory, filename)
        figure.savefig(
            output_file,
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot external position measurements, SITL EKF, and SIMSTATE CSV logs."
    )
    parser.add_argument("--position-measurement", metavar="CSV")
    parser.add_argument("--sitl-ekf-state", metavar="CSV")
    parser.add_argument("--sitl-simstate", metavar="CSV")
    args = parser.parse_args(argv)

    plot_config = SitlEkfPlotterConfig()
    geographic_config = GeographicTransformConfig()

    if not os.path.isdir(plot_config.output_directory):
        raise FileNotFoundError(
            f"Plot output directory does not exist: {plot_config.output_directory}"
        )

    if any((args.position_measurement, args.sitl_ekf_state, args.sitl_simstate)):
        position_path = args.position_measurement
        ekf_path = args.sitl_ekf_state
        simstate_path = args.sitl_simstate
    else:
        logger_config = SitlLoggerConfig()
        position_config = PositionMeasurementLoggerConfig()
        position_path = position_config.output_path
        ekf_path = logger_config.ekf_output_path
        simstate_path = logger_config.simstate_output_path

    plotter = EKFPlotter(plot_config, geographic_config)
    plotter.load_csv_files(position_path, ekf_path, simstate_path)
    plotter.print_alignment_diagnostics()
    plotter.position_vs_time()
    plotter.topdown_trajectory()

    position = plotter._usable_position()
    if position is not None and not position.empty and plotter.sitl_simstate is not None:
        plotter.errors_vs_time()
        plotter.error_distributions()
    else:
        print(
            "Skipping error plots: external measurements and SIMSTATE are required."
        )


if __name__ == "__main__":
    main()


