"""Plot external position measurements, ArduPilot EKF, and SIMSTATE logs.

Position and trajectory plots preserve each source's native samples and use
host_time_ns on a shared time axis. For error plots, external measurement times
define the comparison timeline; EKF and SIMSTATE samples are matched to each
external timestamp within the configured nearest-time tolerance. Coordinates
are NED: x=north, y=east, z=down. SIMSTATE is horizontal only.
"""

import argparse
import math
import os
from dataclasses import dataclass

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


class EKFPlotter:
    def __init__(
        self,
        config: SitlEkfPlotterConfig,
        geographic_config: GeographicTransformConfig,
    ) -> None:
        self.config = config
        self.geographic_config = geographic_config

        if config.sync_tolerance_ms is not None and config.sync_tolerance_ms < 0:
            raise RuntimeError("sync_tolerance_ms must be >= 0 or None")
        if config.figure_width <= 0 or config.figure_height <= 0:
            raise RuntimeError("Figure width and height must be greater than zero")
        if config.dpi <= 0:
            raise RuntimeError("dpi must be greater than zero")
        if geographic_config.earth_radius_m <= 0:
            raise RuntimeError("earth_radius_m must be greater than zero")

        self.position_measurement: pd.DataFrame | None = None
        self.sitl_ekf_state: pd.DataFrame | None = None
        self.sitl_simstate: pd.DataFrame | None = None

    def load_csv_files(
        self,
        position_measurement: str | None = None,
        sitl_ekf_state: str | None = None,
        sitl_simstate: str | None = None,
    ) -> "EKFPlotter":
        """Load any subset of the CSV files. Omitted files are unavailable."""
        self.position_measurement = (
            self._prepare_position(self._read_csv(position_measurement, "Position measurement"))
            if position_measurement else None
        )
        self.sitl_ekf_state = (
            self._prepare_ekf(self._read_csv(sitl_ekf_state, "SITL EKF"))
            if sitl_ekf_state else None
        )
        self.sitl_simstate = (
            self._prepare_simstate(self._read_csv(sitl_simstate, "SITL SIMSTATE"))
            if sitl_simstate else None
        )
        return self

    def plot1(self) -> tuple[Figure, np.ndarray]:
        """Save north/east/down time series at each source's native sample rate.

        All sources share the same time axis (seconds since the first usable
        external measurement), but EKF and SIMSTATE are not downsampled to the
        external measurement rate. This keeps the EKF's full-rate response and
        noise visible. Strict timestamp matching is still used for error plots.
        """
        self._require_data("plot1")
        fig, axes = plt.subplots(
            3, 1,
            figsize=(self.config.figure_width, self.config.figure_height),
            sharex=True,
        )
        position = self._usable_position()
        ekf = self._usable_ekf()
        simstate = self.sitl_simstate

        for axis, component, label in zip(
            axes,
            ("north_m", "east_m", "down_m"),
            ("North (m)", "East (m)", "Down (m)"),
        ):
            if position is not None and not position.empty:
                frame = position.dropna(subset=[component])
                if not frame.empty:
                    axis.plot(
                        self._time_seconds(self._column_series(frame, "host_time_ns")),
                        self._column_series(frame, component),
                        linestyle="none", marker="o", markersize=3,
                        label="External position measurement",
                    )

            if ekf is not None and not ekf.empty:
                frame = ekf.dropna(subset=[component])
                if not frame.empty:
                    axis.plot(
                        self._time_seconds(self._column_series(frame, "host_time_ns")),
                        self._column_series(frame, component),
                        label="EKF (native samples)",
                    )

            if component != "down_m" and simstate is not None and not simstate.empty:
                frame = simstate.dropna(subset=[component])
                if not frame.empty:
                    axis.plot(
                        self._time_seconds(self._column_series(frame, "host_time_ns")),
                        self._column_series(frame, component),
                        label="SIMSTATE truth (native samples)",
                    )

            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            if axis.lines:
                axis.legend()
            else:
                axis.text(
                    0.5, 0.5, "No data available", ha="center", va="center",
                    transform=axis.transAxes,
                )

        axes[-1].set_xlabel(
            "Time since first external measurement (s)"
            if position is not None and not position.empty
            else "Time since first loaded sample (s)"
        )
        fig.suptitle("Position vs time (native sample rates)")
        fig.tight_layout()
        self._save_figure(fig, "plot1_positions_vs_time.png")
        return fig, axes

    def plot2(self) -> tuple[Figure, np.ndarray]:
        """Plot external-to-truth and EKF-to-truth errors on measurement time."""
        self._require_simstate("plot2")
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
        timestamps = self._time_seconds(self._column_series(aligned, "host_time_ns"))
        sources = (
            ("External measurement − SIMSTATE", "measurement", True),
            ("EKF − SIMSTATE", "ekf", False),
        )

        for name, prefix, is_measurement in sources:
            for axis, (component_error, _) in zip(axes, panels):
                column = f"{prefix}_{component_error}"
                if not self._has_finite_values(aligned, column):
                    continue
                if is_measurement:
                    axis.plot(
                        timestamps,
                        self._column_series(aligned, column),
                        linestyle="none", marker="o", markersize=3,
                        label=name,
                    )
                else:
                    axis.plot(
                        timestamps,
                        self._column_series(aligned, column),
                        label=name,
                    )

        for axis, (_, ylabel) in zip(axes, panels):
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if axis.lines:
                axis.legend()
            else:
                axis.text(
                    0.5, 0.5, "No samples matched the time tolerance",
                    ha="center", va="center", transform=axis.transAxes,
                )

        axes[-1].set_xlabel("Time since first external measurement (s)")
        fig.suptitle("Position and EKF errors relative to SIMSTATE truth")
        fig.tight_layout()
        self._save_figure(fig, "plot2_errors_vs_time.png")
        return fig, axes

    def plot3_topdown_trajectory(self) -> tuple[Figure, Axes]:
        """Plot every source's native samples without timestamp matching.

        A trajectory plot compares paths in the same NED coordinate frame, so
        there is no need to downsample the EKF or SIMSTATE to the external
        measurement timestamps.
        """
        self._require_data("plot3_topdown_trajectory")
        fig, axis = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height * 0.8)
        )

        position = self._usable_position()
        if position is not None and not position.empty:
            axis.plot(
                self._column_series(position, "east_m"),
                self._column_series(position, "north_m"),
                linestyle="none", marker="o", markersize=3,
                label="External position measurement",
            )

        ekf = self._usable_ekf()
        if ekf is not None and not ekf.empty:
            axis.plot(
                self._column_series(ekf, "east_m"),
                self._column_series(ekf, "north_m"),
                label="EKF (all samples)",
            )

        simstate = self.sitl_simstate
        if simstate is not None and not simstate.empty:
            truth = simstate.dropna(subset=["east_m", "north_m"])
            if not truth.empty:
                coordinates = truth.loc[:, ["east_m", "north_m"]].astype(float)
                finite_mask = np.isfinite(coordinates).all(axis=1)
                truth = truth.loc[finite_mask, :]
            if not truth.empty:
                axis.plot(
                    self._column_series(truth, "east_m"),
                    self._column_series(truth, "north_m"),
                    label="SIMSTATE truth (all samples)",
                )

        if not axis.lines:
            raise RuntimeError("No valid samples are available for the trajectory plot")

        axis.set_title("Top-down trajectory (all native samples)")
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        self._save_figure(fig, "plot3_topdown_trajectory.png")
        return fig, axis

    def plot4_error_distributions(self) -> tuple[Figure, np.ndarray]:
        """Plot error histograms for external measurements and EKF vs SIMSTATE."""
        self._require_simstate("plot4_error_distributions")
        aligned = self._align_to_position_timeline()

        fig, axes = plt.subplots(
            1, 3,
            figsize=(self.config.figure_width * 1.35, self.config.figure_height * 0.65),
        )
        panels = (
            ("north_error_m", "North error (m)"),
            ("east_error_m", "East error (m)"),
            ("horizontal_error_m", "2D horizontal error (m)"),
        )
        sources = (
            ("External measurement − SIMSTATE", "measurement"),
            ("EKF − SIMSTATE", "ekf"),
        )

        for axis, (component_error, title) in zip(axes, panels):
            series: list[tuple[str, np.ndarray]] = []
            for name, prefix in sources:
                column = f"{prefix}_{component_error}"
                if column not in aligned.columns:
                    continue
                values = pd.to_numeric(
                    self._column_series(aligned, column), errors="coerce"
                ).to_numpy(dtype=float, na_value=np.nan) # type: ignore[attr-defined]

                values = values[np.isfinite(values)]
                if values.size:
                    series.append((name, values))

            if not series:
                axis.text(
                    0.5, 0.5, "No samples matched the time tolerance",
                    ha="center", va="center", transform=axis.transAxes,
                )
            else:
                combined = np.concatenate([values for _, values in series])
                if np.min(combined) == np.max(combined):
                    center = float(combined[0])
                    edges = np.linspace(center - 0.05, center + 0.05, 11)
                else:
                    edges = np.histogram_bin_edges(combined, bins=self.config.histogram_bins)
                for name, values in series:
                    axis.hist(values, bins=edges, density=True, alpha=0.5, label=name)
                axis.legend()

            axis.set_title(title)
            axis.set_xlabel(title)
            axis.set_ylabel("Probability density")
            axis.grid(True, alpha=0.25)

        fig.suptitle("Position and EKF error distributions relative to SIMSTATE")
        fig.tight_layout()
        self._save_figure(fig, "plot4_error_distributions.png")
        return fig, axes

    @staticmethod
    def _column_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
        """Return one CSV column as a Series, rejecting duplicate column names."""
        column = frame.loc[:, column_name]
        if not isinstance(column, pd.Series):
            raise RuntimeError(f"CSV contains duplicate column name: {column_name}")
        return column

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
        """Return a numeric CSV column as a Series, coercing invalid values to NaN."""
        column = pd.to_numeric(
            EKFPlotter._column_series(frame, column_name), errors="coerce"
        )
        if not isinstance(column, pd.Series):
            raise RuntimeError(f"CSV column could not be converted to numeric: {column_name}")
        return column

    @staticmethod
    def _read_csv(filename: str, description: str) -> pd.DataFrame:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{description} CSV does not exist: {filename}")
        frame = pd.read_csv(filename)
        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.empty:
            raise RuntimeError(f"{description} CSV is empty: {filename}")
        return frame

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        description: str,
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise RuntimeError(
                f"{description} CSV is missing required columns: {', '.join(missing)}"
            )

    @classmethod
    def _prepare_host_time(
        cls: type["EKFPlotter"],
        frame: pd.DataFrame,
        description: str,
    ) -> pd.DataFrame:
        cls._require_columns(frame, ("host_time_ns",), description)
        result = frame.copy()
        numeric_timestamps = pd.to_numeric(
            cls._column_series(result, "host_time_ns"), errors="coerce"
        )
        if not isinstance(numeric_timestamps, pd.Series):
            raise RuntimeError(f"Could not read {description} host_time_ns as a CSV column")
        result["host_time_ns"] = numeric_timestamps
        result = result.dropna(subset=["host_time_ns"])
        if result.empty:
            raise RuntimeError(f"{description} CSV has no valid host_time_ns values")
        result["host_time_ns"] = result["host_time_ns"].astype("int64")
        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _prepare_position(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_host_time(frame, "Position measurement")
        self._require_columns(result, ("x", "y", "z"), "Position measurement")
        if "frame" in result.columns:
            frames = {str(value).strip().upper() for value in result["frame"].dropna()}
            if frames - {"NED"}:
                raise RuntimeError(
                    "Position-measurement CSV must be transformed to NED before plotting; "
                    f"found frame value(s): {', '.join(sorted(frames))}"
                )

        result["north_m"] = self._numeric_column(result, "x")
        result["east_m"] = self._numeric_column(result, "y")
        result["down_m"] = self._numeric_column(result, "z")
        if "valid" in result.columns:
            result["_valid_bool"] = result["valid"].map(self._parse_bool)
        else:
            result["_valid_bool"] = True
        return result

    def _prepare_ekf(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_host_time(frame, "SITL EKF")
        if all(column in result.columns for column in ("x_north_m", "y_east_m", "z_down_m")):
            columns = ("x_north_m", "y_east_m", "z_down_m")
        elif all(column in result.columns for column in ("x", "y", "z")):
            columns = ("x", "y", "z")
        else:
            raise RuntimeError(
                "SITL EKF CSV must contain x_north_m/y_east_m/z_down_m or x/y/z"
            )
        result["north_m"] = self._numeric_column(result, columns[0])
        result["east_m"] = self._numeric_column(result, columns[1])
        result["down_m"] = self._numeric_column(result, columns[2])
        # Keep the autopilot's own timestamp available for diagnostics. It is
        # not directly comparable to host_time_ns until a clock mapping exists.
        if "sitl_time_ms" in result.columns:
            result["sitl_time_ms"] = self._numeric_column(result, "sitl_time_ms")
        return result

    def _prepare_simstate(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._prepare_host_time(frame, "SITL SIMSTATE")
        self._require_columns(result, ("latitude_deg", "longitude_deg"), "SITL SIMSTATE")
        result["latitude_deg"] = self._numeric_column(result, "latitude_deg")
        result["longitude_deg"] = self._numeric_column(result, "longitude_deg")
        result = result.dropna(subset=["latitude_deg", "longitude_deg"])
        finite_mask = (
            np.isfinite(self._column_series(result, "latitude_deg").astype(float))
            & np.isfinite(self._column_series(result, "longitude_deg").astype(float))
        )
        result = result.loc[finite_mask, :]
        if result.empty:
            raise RuntimeError("SITL SIMSTATE CSV has no finite latitude/longitude rows")
        latitude = self._column_series(result, "latitude_deg")
        longitude = self._column_series(result, "longitude_deg")
        if not latitude.between(-90, 90).all():
            raise RuntimeError("SITL SIMSTATE CSV contains an invalid latitude")
        if not longitude.between(-180, 180).all():
            raise RuntimeError("SITL SIMSTATE CSV contains an invalid longitude")

        radius = self.geographic_config.earth_radius_m
        latitude0 = self.geographic_config.origin_latitude_deg
        longitude0 = self.geographic_config.origin_longitude_deg
        latitude0_rad = math.radians(latitude0)
        delta_latitude = np.radians(result["latitude_deg"] - latitude0)
        delta_longitude_deg = (
            (result["longitude_deg"] - longitude0 + 180.0) % 360.0
        ) - 180.0
        delta_longitude = np.radians(delta_longitude_deg)
        result["north_m"] = radius * delta_latitude
        result["east_m"] = radius * math.cos(latitude0_rad) * delta_longitude
        return result.reset_index(drop=True)

    @staticmethod
    def _parse_bool(value: object) -> bool:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value) and bool(np.isfinite(value))
        return str(value).strip().lower() in {"true", "t", "yes", "y", "1", "valid"}

    def _usable_position(self) -> pd.DataFrame | None:
        if self.position_measurement is None:
            return None
        result = self.position_measurement
        if not self.config.include_invalid_measurements:
            valid_mask = self._column_series(result, "_valid_bool")
            result = result.loc[valid_mask, :]
        columns = ["north_m", "east_m", "down_m"]
        result = result.dropna(subset=columns)
        if not result.empty:
            coordinates = result.loc[:, columns].astype(float)
            finite_mask = np.isfinite(coordinates).all(axis=1)
            result = result.loc[finite_mask, :]
        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _usable_ekf(self) -> pd.DataFrame | None:
        if self.sitl_ekf_state is None:
            return None
        result = self.sitl_ekf_state.dropna(subset=["north_m", "east_m", "down_m"])
        if not result.empty:
            columns = ["north_m", "east_m", "down_m"]
            coordinates = result.loc[:, columns].astype(float)
            finite_mask = np.isfinite(coordinates).all(axis=1)
            result = result.loc[finite_mask, :]
        return result.sort_values("host_time_ns").reset_index(drop=True)

    def _require_simstate(self, method_name: str) -> None:
        if self.sitl_simstate is None:
            raise RuntimeError(
                f"{method_name} requires the SIMSTATE CSV for horizontal ground truth"
            )

    def _require_data(self, method_name: str) -> None:
        if not any((
            self.position_measurement is not None,
            self.sitl_ekf_state is not None,
            self.sitl_simstate is not None,
        )):
            raise RuntimeError(f"Load at least one CSV file before calling {method_name}")

    @staticmethod
    def _has_finite_values(frame: pd.DataFrame, column_name: str) -> bool:
        """Return whether a numeric column contains at least one finite value."""
        if column_name not in frame.columns:
            return False
        values = pd.to_numeric(
            EKFPlotter._column_series(frame, column_name), errors="coerce"
        )
        if not isinstance(values, pd.Series):
            raise RuntimeError(f"Could not read numeric CSV column: {column_name}")
        return bool(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).any())

    def _sync_tolerance_ns(self) -> int | None:
        """Convert configured matching tolerance from milliseconds to nanoseconds."""
        if self.config.sync_tolerance_ms is None:
            return None
        return int(round(self.config.sync_tolerance_ms * 1_000_000))

    def _align_to_position_timeline(
        self,
        position: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Match EKF and SIMSTATE samples independently to each external sample.

        The returned ``host_time_ns`` always comes from the position-measurement
        CSV. Source timestamps are retained separately so sample offsets can be
        inspected. An unmatched source remains NaN and therefore appears as a gap
        rather than being silently compared with a distant sample.
        """
        if position is None:
            position = self._usable_position()
        if position is None or position.empty:
            raise RuntimeError(
                "External position measurements are required as the common timeline"
            )

        aligned = pd.DataFrame(
            {
                "host_time_ns": self._column_series(position, "host_time_ns").astype("int64"),
                "measurement_north_m": self._column_series(position, "north_m"),
                "measurement_east_m": self._column_series(position, "east_m"),
                "measurement_down_m": self._column_series(position, "down_m"),
            }
        ).sort_values("host_time_ns").reset_index(drop=True)
        tolerance_ns = self._sync_tolerance_ns()

        ekf = self._usable_ekf()
        if ekf is not None and not ekf.empty:
            ekf_samples = pd.DataFrame(
                {
                    "ekf_host_time_ns": self._column_series(ekf, "host_time_ns").astype("int64"),
                    "ekf_north_m": self._column_series(ekf, "north_m"),
                    "ekf_east_m": self._column_series(ekf, "east_m"),
                    "ekf_down_m": self._column_series(ekf, "down_m"),
                }
            ).sort_values("ekf_host_time_ns")
            aligned = pd.merge_asof(
                aligned.sort_values("host_time_ns"),
                ekf_samples,
                left_on="host_time_ns",
                right_on="ekf_host_time_ns",
                direction="nearest",
                tolerance=tolerance_ns,
            )

        if self.sitl_simstate is not None and not self.sitl_simstate.empty:
            truth_source = self.sitl_simstate.dropna(subset=["north_m", "east_m"])
            if not truth_source.empty:
                truth_coordinates = truth_source.loc[:, ["north_m", "east_m"]].astype(float)
                finite_mask = np.isfinite(truth_coordinates).all(axis=1)
                truth_source = truth_source.loc[finite_mask, :]
            if not truth_source.empty:
                truth_samples = pd.DataFrame(
                    {
                        "simstate_host_time_ns": self._column_series(
                            truth_source, "host_time_ns"
                        ).astype("int64"),
                        "truth_north_m": self._column_series(truth_source, "north_m"),
                        "truth_east_m": self._column_series(truth_source, "east_m"),
                    }
                ).sort_values("simstate_host_time_ns")
                aligned = pd.merge_asof(
                    aligned.sort_values("host_time_ns"),
                    truth_samples,
                    left_on="host_time_ns",
                    right_on="simstate_host_time_ns",
                    direction="nearest",
                    tolerance=tolerance_ns,
                )

        if "truth_north_m" in aligned.columns and "truth_east_m" in aligned.columns:
            aligned["measurement_north_error_m"] = (
                aligned["measurement_north_m"] - aligned["truth_north_m"]
            )
            aligned["measurement_east_error_m"] = (
                aligned["measurement_east_m"] - aligned["truth_east_m"]
            )
            aligned["measurement_horizontal_error_m"] = np.sqrt(
                aligned["measurement_north_error_m"] ** 2
                + aligned["measurement_east_error_m"] ** 2
            )

            if "ekf_north_m" in aligned.columns and "ekf_east_m" in aligned.columns:
                aligned["ekf_north_error_m"] = aligned["ekf_north_m"] - aligned["truth_north_m"]
                aligned["ekf_east_error_m"] = aligned["ekf_east_m"] - aligned["truth_east_m"]
                aligned["ekf_horizontal_error_m"] = np.sqrt(
                    aligned["ekf_north_error_m"] ** 2 + aligned["ekf_east_error_m"] ** 2
                )

        return aligned.reset_index(drop=True)

    def print_alignment_diagnostics(self) -> None:
        """Print match counts and timestamp offsets for the error-plot timeline."""
        position = self._usable_position()
        if position is None or position.empty:
            print("Alignment diagnostics unavailable: no usable external measurements.")
            return

        aligned = self._align_to_position_timeline(position)
        total = len(aligned)
        print(f"Alignment to external measurements: {total} timeline sample(s).")

        sources = (
            ("EKF", "ekf_host_time_ns", "ekf_north_m"),
            ("SIMSTATE", "simstate_host_time_ns", "truth_north_m"),
        )
        host_times = self._numeric_column(aligned, "host_time_ns")

        for name, source_time_column, value_column in sources:
            if source_time_column not in aligned.columns or value_column not in aligned.columns:
                print(f"  {name}: no data available to match.")
                continue

            source_times = pd.to_numeric(
                self._column_series(aligned, source_time_column), errors="coerce"
            )
            source_values = pd.to_numeric(
                self._column_series(aligned, value_column), errors="coerce"
            )
            matched_mask = source_times.notna() & source_values.notna() # type: ignore[attr-defined]
            matched_count = int(matched_mask.sum())
            print(f"  {name}: matched {matched_count}/{total} samples.")

            if matched_count:
                offsets_ms = (
                    source_times.loc[matched_mask].astype(float) # type: ignore[attr-defined]

                    - host_times.loc[matched_mask].astype(float)
                ) / 1_000_000.0
                offset_values = offsets_ms.to_numpy(dtype=float, na_value=np.nan)
                offset_values = offset_values[np.isfinite(offset_values)]
                if offset_values.size:
                    print(
                        f"    source-minus-measurement offset (ms): "
                        f"min={np.min(offset_values):.2f}, "
                        f"median={np.median(offset_values):.2f}, "
                        f"max={np.max(offset_values):.2f}"
                    )

        print(
            f"  Configured matching tolerance: "
            f"{self.config.sync_tolerance_ms} ms"
        )

    def _time_origin_ns(self) -> int:
        position = self._usable_position()
        if position is not None and not position.empty:
            return int(self._column_series(position, "host_time_ns").min())

        starts: list[int] = []
        for frame in (self.sitl_ekf_state, self.sitl_simstate):
            if frame is not None and not frame.empty:
                starts.append(int(self._column_series(frame, "host_time_ns").min()))
        if not starts:
            raise RuntimeError("No loaded CSV contains usable timestamps")
        return min(starts)

    def _time_seconds(self, timestamps: pd.Series) -> pd.Series:
        return (timestamps.astype("int64") - self._time_origin_ns()) / 1_000_000_000.0

    def _save_figure(self, figure: Figure, filename: str) -> None:
        output_directory = self.config.output_directory
        if not os.path.isdir(output_directory):
            raise FileNotFoundError(
                f"Plot output directory does not exist: {output_directory}"
            )
        output_file = os.path.join(output_directory, filename)
        figure.savefig(output_file, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(figure)



@dataclass
class _PlotterArguments:
    position_measurement: str | None
    sitl_ekf_state: str | None
    sitl_simstate: str | None

def main(argv: list[str] | None = None) -> None:
    """Load CSV paths from command-line arguments and save applicable plots.

    With no CSV arguments, the default paths from the logger configurations
    are used. If any CSV argument is provided, only the explicitly supplied
    CSV files are loaded.
    """
    parser = argparse.ArgumentParser(
        description="Plot position-measurement, SITL EKF, and SIMSTATE CSV logs.",
        epilog=(
            "With no CSV path arguments, the configured default CSV files are used. "
            "If any path argument is supplied, only the explicitly supplied files "
            "are loaded. The configured output directory must already exist."
        ),
    )
    parser.add_argument(
        "--position-measurement",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to the position-measurement CSV.",
    )
    parser.add_argument(
        "--sitl-ekf-state",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to the SITL EKF-state CSV.",
    )
    parser.add_argument(
        "--sitl-simstate",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to the SITL SIMSTATE CSV.",
    )
    parsed_args = parser.parse_args(argv)
    args = _PlotterArguments(
        position_measurement=parsed_args.position_measurement,
        sitl_ekf_state=parsed_args.sitl_ekf_state,
        sitl_simstate=parsed_args.sitl_simstate,
    )

    plotter_config = SitlEkfPlotterConfig()
    geographic_config = GeographicTransformConfig()

    supplied_csv_paths = any((
        args.position_measurement is not None,
        args.sitl_ekf_state is not None,
        args.sitl_simstate is not None,
    ))
    if supplied_csv_paths:
        position_measurement_path: str | None = args.position_measurement
        sitl_ekf_state_path: str | None = args.sitl_ekf_state
        sitl_simstate_path: str | None = args.sitl_simstate
    else:
        logger_config = SitlLoggerConfig()
        measurement_config = PositionMeasurementLoggerConfig()
        position_measurement_path = measurement_config.output_path
        sitl_ekf_state_path = logger_config.ekf_output_path
        sitl_simstate_path = logger_config.simstate_output_path

    if not os.path.isdir(plotter_config.output_directory):
        raise FileNotFoundError(
            f"Plot output directory does not exist: {plotter_config.output_directory}"
        )

    plotter = EKFPlotter(
        config=plotter_config,
        geographic_config=geographic_config,
    )
    plotter.load_csv_files(
        position_measurement=position_measurement_path,
        sitl_ekf_state=sitl_ekf_state_path,
        sitl_simstate=sitl_simstate_path,
    )

    # Report whether the error-plot data can be matched to external timestamps.
    plotter.print_alignment_diagnostics()

    # Position-versus-time and trajectory plots work with any loaded source.
    plotter.plot1()
    plotter.plot3_topdown_trajectory()

    # Error plots use the external-position timestamps as the common timeline.
    position = plotter._usable_position()
    if plotter.sitl_simstate is not None and position is not None and not position.empty:
        plotter.plot2()
        plotter.plot4_error_distributions()
    else:
        print(
            "Skipping plot2 and plot4: these plots require usable external position "
            "measurements (the common timeline) and SIMSTATE ground truth."
        )


if __name__ == "__main__":
    main()

