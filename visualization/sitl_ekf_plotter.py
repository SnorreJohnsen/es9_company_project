"""Plot and compare external position measurements, ArduPilot EKF, and SITL truth.

Single-run mode remains available. Batch mode expects:

    experiment_root/
        baseline/
            <position CSV>   <EKF CSV>   <SIMSTATE CSV>
        EK3_POSNE_M_NSE/
            0.5x/
                <position CSV>   <EKF CSV>   <SIMSTATE CSV>
            1.5x/
                ...
        EK3_ACC_P_NSE/
            0.5x/
                ...

The CSV filenames are taken from the project's logger config by default. They
can also be provided explicitly on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """Load and plot one simulation run."""

    def __init__(
        self,
        config: SitlEkfPlotterConfig,
        geographic_config: GeographicTransformConfig,
        output_directory: str | Path | None = None,
    ) -> None:
        self.config = config
        self.geographic_config = geographic_config
        self.geographic_transformer = GeographicToNedTransformer(geographic_config)
        self.output_directory = Path(
            output_directory if output_directory is not None else config.output_directory
        )

        self.position_measurement: pd.DataFrame | None = None
        self.sitl_ekf_state: pd.DataFrame | None = None
        self.sitl_simstate: pd.DataFrame | None = None

    def load_csv_files(
        self,
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
        """Plot North, East, and Down at each source's native sample rate."""
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
                    linestyle="none", marker="o", markersize=3,
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
        """Plot external measurement and EKF errors relative to SIMSTATE."""
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
            for prefix, source_label in (
                ("measurement", "External measurement − SIMSTATE"),
                ("ekf", "EKF − SIMSTATE"),
            ):
                column = f"{prefix}_{error_name}"
                if column not in aligned.columns:
                    continue
                values = self._numeric(aligned, column)
                finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
                if finite.any():
                    axis.plot(time_s, values, label=source_label)
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
        """Plot all available North/East trajectories."""
        self._require_data()
        fig, axis = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height * 0.8),
        )
        position = self._usable_position()
        if position is not None and not position.empty:
            axis.plot(
                self._series(position, "east_m"),
                self._series(position, "north_m"),
                linestyle="none", marker="o", markersize=3,
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
            for prefix, label in (
                ("measurement", "External measurement − SIMSTATE"),
                ("ekf", "EKF − SIMSTATE"),
            ):
                column = f"{prefix}_{error_name}"
                if column not in aligned.columns:
                    continue
                values = self._numeric(aligned, column).to_numpy(
                    dtype=float, na_value=np.nan,
                )
                values = values[np.isfinite(values)]
                if values.size:
                    histograms.append((label, values))

            if histograms:
                combined = np.concatenate([values for _, values in histograms])
                if np.min(combined) == np.max(combined):
                    center = float(combined[0])
                    bins = np.linspace(center - 0.05, center + 0.05, 11)
                else:
                    bins = np.histogram_bin_edges(
                        combined, bins=self.config.histogram_bins,
                    )
                for label, values in histograms:
                    axis.hist(values, bins=bins, density=True, alpha=0.5, label=label)
                axis.legend()
            else:
                axis.text(
                    0.5, 0.5, "No matched samples", ha="center", va="center",
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
    def _numeric(cls, frame: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(cls._series(frame, column), errors="coerce")
        if not isinstance(values, pd.Series):
            raise RuntimeError(f"Could not read numeric CSV column: {column}")
        return values

    @classmethod
    def _prepare_time(cls, frame: pd.DataFrame, name: str) -> pd.DataFrame:
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
        missing = [column for column in ("x", "y", "z") if column not in result.columns]
        if missing:
            raise RuntimeError(f"Position CSV missing columns: {', '.join(missing)}")
        if "frame" in result.columns:
            frames = {str(value).strip().upper() for value in result["frame"].dropna()}
            if frames - {"NED"}:
                raise RuntimeError("Position measurements must be transformed to NED first")
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
            "x_north_m", "y_east_m", "z_down_m",
        )):
            north, east, down = "x_north_m", "y_east_m", "z_down_m"
        elif all(column in result.columns for column in ("x", "y", "z")):
            north, east, down = "x", "y", "z"
        else:
            raise RuntimeError("EKF CSV needs x_north_m/y_east_m/z_down_m or x/y/z")
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
            raise RuntimeError(f"SIMSTATE CSV missing columns: {', '.join(missing)}")
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
            "true", "t", "yes", "y", "1", "valid",
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
        result = self.sitl_ekf_state.dropna(subset=["north_m", "east_m", "down_m"])
        if not result.empty:
            columns = ["north_m", "east_m", "down_m"]
            finite = np.isfinite(result[columns].astype(float)).all(axis=1)
            result = result.loc[finite, :]
        return result.sort_values("host_time_ns").reset_index(drop=True)

    @staticmethod
    def _interpolate_to_timeline(
        source: pd.DataFrame,
        target_times_ns: np.ndarray,
        value_columns: tuple[str, ...],
        output_prefix: str,
        max_bracket_distance_ns: int | None,
    ) -> pd.DataFrame:
        """Linearly interpolate source data without extrapolating.

        When a sync tolerance is configured, both bracketing source samples must
        be within that tolerance of the target timestamp. This avoids silently
        interpolating across large data gaps.
        """
        clean = source.dropna(subset=["host_time_ns", *value_columns]).copy()
        clean = clean.sort_values("host_time_ns").drop_duplicates(
            subset=["host_time_ns"], keep="last",
        )
        target_times_ns = np.asarray(target_times_ns, dtype=np.int64)
        output = pd.DataFrame(index=np.arange(len(target_times_ns)))
        if clean.empty:
            for column in value_columns:
                output[f"{output_prefix}_{column}"] = np.nan
            output[f"_{output_prefix}_matched"] = False
            return output

        source_times = clean["host_time_ns"].to_numpy(dtype=np.int64)
        right = np.searchsorted(source_times, target_times_ns, side="left")
        in_range = (target_times_ns >= source_times[0]) & (target_times_ns <= source_times[-1])
        safe_right = np.clip(right, 0, len(source_times) - 1)
        exact = in_range & (source_times[safe_right] == target_times_ns)
        left = np.where(exact, safe_right, np.clip(right - 1, 0, len(source_times) - 1))
        right = np.where(exact, safe_right, safe_right)

        matched = in_range.copy()
        if max_bracket_distance_ns is not None:
            left_distance = target_times_ns - source_times[left]
            right_distance = source_times[right] - target_times_ns
            matched &= (left_distance <= max_bracket_distance_ns)
            matched &= (right_distance <= max_bracket_distance_ns)

        for column in value_columns:
            values = pd.to_numeric(clean[column], errors="coerce").to_numpy(dtype=float)
            interpolated = np.interp(target_times_ns, source_times, values)
            interpolated[~matched] = np.nan
            output[f"{output_prefix}_{column}"] = interpolated
        output[f"_{output_prefix}_matched"] = matched
        return output

    def _align_to_position_timeline(self) -> pd.DataFrame:
        """Interpolate EKF and SIMSTATE onto the external-measurement timeline."""
        position = self._usable_position()
        if position is None or position.empty:
            raise RuntimeError("Error plots require external position measurements")
        if self.sitl_simstate is None or self.sitl_simstate.empty:
            raise RuntimeError("Error plots require SIMSTATE ground truth")

        target_times = self._series(position, "host_time_ns").to_numpy(dtype=np.int64)
        aligned = pd.DataFrame({
            "host_time_ns": target_times,
            "measurement_north_m": self._series(position, "north_m").to_numpy(),
            "measurement_east_m": self._series(position, "east_m").to_numpy(),
            "measurement_down_m": self._series(position, "down_m").to_numpy(),
        })
        tolerance_ns = (
            None if self.config.sync_tolerance_ms is None
            else int(self.config.sync_tolerance_ms * 1_000_000)
        )

        ekf = self._usable_ekf()
        if ekf is not None and not ekf.empty:
            ekf_values = self._interpolate_to_timeline(
                ekf, target_times,
                ("north_m", "east_m", "down_m"),
                "ekf", tolerance_ns,
            )
            aligned = pd.concat([aligned, ekf_values], axis=1)
        else:
            aligned["_ekf_matched"] = False

        truth_values = self._interpolate_to_timeline(
            self.sitl_simstate, target_times,
            ("north_m", "east_m"), "truth", tolerance_ns,
        )
        aligned = pd.concat([aligned, truth_values], axis=1)

        sources = (
            ("measurement", "measurement_north_m", "measurement_east_m"),
            ("ekf", "ekf_north_m", "ekf_east_m"),
        )
        for prefix, north, east in sources:
            if north not in aligned.columns or east not in aligned.columns:
                continue
            aligned[f"{prefix}_north_error_m"] = aligned[north] - aligned["truth_north_m"]
            aligned[f"{prefix}_east_error_m"] = aligned[east] - aligned["truth_east_m"]
            aligned[f"{prefix}_horizontal_error_m"] = np.sqrt(
                aligned[f"{prefix}_north_error_m"] ** 2
                + aligned[f"{prefix}_east_error_m"] ** 2
            )
        return aligned.reset_index(drop=True)

    def print_alignment_diagnostics(self) -> None:
        if self.position_measurement is None or self.sitl_simstate is None:
            return
        aligned = self._align_to_position_timeline()
        total = len(aligned)
        print(f"Alignment to external-measurement timeline: {total} samples.")
        for label, column in (
            ("EKF", "_ekf_matched"),
            ("SIMSTATE", "_truth_matched"),
        ):
            if column not in aligned.columns:
                print(f"  {label}: no samples available.")
                continue
            matched = int(aligned[column].sum())
            print(f"  {label}: interpolated/matched {matched}/{total} samples.")
        print(
            "  Maximum distance from each target timestamp to either interpolation "
            f"bracket: {self.config.sync_tolerance_ms} ms (per side)."
            if self.config.sync_tolerance_ms is not None
            else "  Interpolation tolerance: unlimited within source time range."
        )

    def _require_data(self) -> None:
        if all(data is None for data in (
            self.position_measurement, self.sitl_ekf_state, self.sitl_simstate,
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
        return (timestamps.astype("int64") - self._time_origin_ns()) / 1_000_000_000.0

    def _save_figure(self, figure: Figure, filename: str) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            self.output_directory / filename,
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


@dataclass(frozen=True)
class ExperimentSpec:
    """One baseline or parameter-factor simulation directory."""

    name: str
    parameter_name: str
    factor_label: str
    factor_value: float | None
    directory: Path
    is_baseline: bool = False


class BatchExperimentAnalyzer:
    """Discover, plot, and quantitatively compare multiple simulation runs."""

    def __init__(
        self,
        plot_config: SitlEkfPlotterConfig,
        geographic_config: GeographicTransformConfig,
        position_filename: str,
        ekf_filename: str,
        simstate_filename: str,
        baseline_directory_name: str = "baseline",
    ) -> None:
        self.plot_config = plot_config
        self.geographic_config = geographic_config
        self.position_filename = Path(position_filename).name
        self.ekf_filename = Path(ekf_filename).name
        self.simstate_filename = Path(simstate_filename).name
        self.baseline_directory_name = baseline_directory_name

    @staticmethod
    def _read_metadata(folder: Path) -> dict[str, Any]:
        metadata_path = folder / "metadata.json"
        if not metadata_path.is_file():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            if not isinstance(metadata, dict):
                raise ValueError("metadata.json must contain a JSON object")
            return metadata
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"WARNING: Could not read {metadata_path}: {error}")
            return {}

    @staticmethod
    def _parse_factor(folder_name: str, metadata: dict[str, Any]) -> tuple[str, float | None]:
        metadata_factor = metadata.get("factor", metadata.get("factor_compared_to_baseline"))
        if metadata_factor is not None:
            try:
                value = float(metadata_factor)
                return str(metadata_factor), value
            except (TypeError, ValueError):
                return str(metadata_factor), None

        # Examples accepted: 0.5, 0.5x, factor_0.5, 1p5x.
        normalised = folder_name.replace("p", ".")
        match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", normalised)
        if match:
            try:
                return folder_name, float(match.group(0))
            except ValueError:
                pass
        return folder_name, None

    def discover_experiments(self, root_directory: str | Path) -> list[ExperimentSpec]:
        root = Path(root_directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Experiment root does not exist: {root}")

        baseline_path = root / self.baseline_directory_name
        if not baseline_path.is_dir():
            raise FileNotFoundError(
                f"Baseline directory not found: {baseline_path}. "
                "Create it or pass --baseline-dir with its directory name."
            )

        experiments = [ExperimentSpec(
            name="baseline",
            parameter_name="baseline",
            factor_label="1.0x",
            factor_value=1.0,
            directory=baseline_path,
            is_baseline=True,
        )]

        ignored_names = {
            self.baseline_directory_name.lower(),
            "comparison_plots", "plots", "__pycache__",
        }
        for parameter_directory in sorted(path for path in root.iterdir() if path.is_dir()):
            if parameter_directory.name.lower() in ignored_names:
                continue
            factor_directories = sorted(path for path in parameter_directory.iterdir() if path.is_dir())
            for factor_directory in factor_directories:
                metadata = self._read_metadata(factor_directory)
                factor_label, factor_value = self._parse_factor(factor_directory.name, metadata)
                parameter_name = str(metadata.get("parameter", parameter_directory.name))
                experiments.append(ExperimentSpec(
                    name=f"{parameter_name} ({factor_label})",
                    parameter_name=parameter_name,
                    factor_label=factor_label,
                    factor_value=factor_value,
                    directory=factor_directory,
                    is_baseline=False,
                ))
        if len(experiments) == 1:
            raise RuntimeError(f"No parameter/factor experiment folders found under {root}")
        return experiments

    @staticmethod
    def _guess_csv_by_role(folder: Path, role: str, expected_name: str) -> Path | None:
        expected = folder / expected_name
        if expected.is_file():
            return expected

        candidates = sorted(folder.glob("*.csv"))
        scored: list[tuple[int, Path]] = []
        for candidate in candidates:
            try:
                columns = {str(column).strip().lower() for column in pd.read_csv(candidate, nrows=2).columns}
            except Exception:
                continue
            filename = candidate.stem.lower()
            score = 0
            if role == "position":
                if {"x", "y", "z"}.issubset(columns):
                    score += 3
                if "position" in filename or "measurement" in filename or "gps" in filename:
                    score += 4
                if "frame" in columns or "valid" in columns:
                    score += 1
            elif role == "ekf":
                if {"x_north_m", "y_east_m", "z_down_m"}.issubset(columns):
                    score += 6
                elif {"x", "y", "z"}.issubset(columns):
                    score += 2
                if "ekf" in filename:
                    score += 5
            elif role == "simstate":
                if {"latitude_deg", "longitude_deg"}.issubset(columns):
                    score += 6
                if "simstate" in filename or "sim_state" in filename:
                    score += 5
            if score:
                scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        if not scored:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise RuntimeError(
                f"Ambiguous {role} CSV files in {folder}: "
                f"{scored[0][1].name} and {scored[1][1].name}. "
                f"Pass --{role}-filename to select one explicitly."
            )
        return scored[0][1]

    def _resolve_csv_paths(self, folder: Path) -> tuple[Path, Path, Path]:
        choices = (
            ("position", self.position_filename),
            ("ekf", self.ekf_filename),
            ("simstate", self.simstate_filename),
        )
        resolved: list[Path] = []
        missing: list[str] = []
        for role, expected_name in choices:
            path = self._guess_csv_by_role(folder, role, expected_name)
            if path is None:
                missing.append(f"{role} (expected {expected_name})")
            else:
                resolved.append(path)
        if missing:
            raise FileNotFoundError(
                f"Missing CSV file(s) in {folder}: {', '.join(missing)}. "
                "CSV files must be directly inside each experiment folder."
            )
        return resolved[0], resolved[1], resolved[2]

    @staticmethod
    def _calculate_error_metrics(aligned: pd.DataFrame, prefix: str) -> dict[str, float | int]:
        north_column = f"{prefix}_north_error_m"
        east_column = f"{prefix}_east_error_m"
        horizontal_column = f"{prefix}_horizontal_error_m"
        if not {north_column, east_column, horizontal_column}.issubset(aligned.columns):
            return {
                f"{prefix}_samples": 0,
                f"{prefix}_horizontal_rmse_m": np.nan,
                f"{prefix}_horizontal_mae_m": np.nan,
                f"{prefix}_horizontal_p95_m": np.nan,
                f"{prefix}_horizontal_max_m": np.nan,
                f"{prefix}_north_bias_m": np.nan,
                f"{prefix}_east_bias_m": np.nan,
            }

        north = pd.to_numeric(aligned[north_column], errors="coerce").to_numpy(dtype=float)
        east = pd.to_numeric(aligned[east_column], errors="coerce").to_numpy(dtype=float)
        horizontal = pd.to_numeric(aligned[horizontal_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(north) & np.isfinite(east) & np.isfinite(horizontal)
        north, east, horizontal = north[valid], east[valid], horizontal[valid]
        if horizontal.size == 0:
            return {
                f"{prefix}_samples": 0,
                f"{prefix}_horizontal_rmse_m": np.nan,
                f"{prefix}_horizontal_mae_m": np.nan,
                f"{prefix}_horizontal_p95_m": np.nan,
                f"{prefix}_horizontal_max_m": np.nan,
                f"{prefix}_north_bias_m": np.nan,
                f"{prefix}_east_bias_m": np.nan,
            }
        return {
            f"{prefix}_samples": int(horizontal.size),
            f"{prefix}_horizontal_rmse_m": float(np.sqrt(np.mean(horizontal ** 2))),
            f"{prefix}_horizontal_mae_m": float(np.mean(horizontal)),
            f"{prefix}_horizontal_p95_m": float(np.percentile(horizontal, 95)),
            f"{prefix}_horizontal_max_m": float(np.max(horizontal)),
            f"{prefix}_north_bias_m": float(np.mean(north)),
            f"{prefix}_east_bias_m": float(np.mean(east)),
        }

    def _analyze_one(
        self,
        experiment: ExperimentSpec,
        generate_plots: bool = True,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        position_path, ekf_path, simstate_path = self._resolve_csv_paths(experiment.directory)
        plot_directory = experiment.directory / "plots"
        plotter = EKFPlotter(
            self.plot_config, self.geographic_config,
            output_directory=plot_directory,
        )
        plotter.load_csv_files(str(position_path), str(ekf_path), str(simstate_path))
        plotter.print_alignment_diagnostics()

        if generate_plots:
            plotter.position_vs_time()
            plotter.topdown_trajectory()
            plotter.errors_vs_time()
            plotter.error_distributions()

        aligned = plotter._align_to_position_timeline()
        measurement_metrics = self._calculate_error_metrics(aligned, "measurement")
        ekf_metrics = self._calculate_error_metrics(aligned, "ekf")
        row: dict[str, Any] = {
            "experiment": experiment.name,
            "parameter": experiment.parameter_name,
            "factor": experiment.factor_value,
            "factor_label": experiment.factor_label,
            "is_baseline": experiment.is_baseline,
            "directory": str(experiment.directory),
            **measurement_metrics,
            **ekf_metrics,
        }

        metadata = self._read_metadata(experiment.directory)
        row["parameter_value"] = metadata.get("parameter_value", np.nan)
        row["baseline_parameter_value"] = metadata.get("baseline_parameter_value", np.nan)
        return row, aligned["ekf_horizontal_error_m"].to_numpy(dtype=float)

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
        return safe or "parameter"

    @staticmethod
    def _write_markdown_summary(results: pd.DataFrame, output_path: Path) -> None:
        columns = [
            "parameter", "factor_label",
            "measurement_horizontal_rmse_m", "ekf_horizontal_rmse_m",
            "improvement_vs_baseline_pct", "improvement_vs_gps_pct",
            "ekf_horizontal_p95_m", "ekf_horizontal_max_m", "ekf_samples",
        ]
        headings = [
            "Parameter", "Factor", "GPS RMSE (m)",
            "EKF RMSE (m)", "vs baseline (%)", "vs GPS (%)",
            "EKF P95 (m)", "EKF max (m)", "Samples",
        ]
        lines = [
            "# EKF parameter experiment comparison", "",
            "Horizontal errors are evaluated against SIMSTATE at the external-measurement timestamps. "
            "EKF and SIMSTATE samples are linearly interpolated to those timestamps; no extrapolation is performed.",
            "",
            "Positive improvement means a lower EKF horizontal RMSE.",
            "",
            "| " + " | ".join(headings) + " |",
            "| " + " | ".join(["---"] * len(headings)) + " |",
        ]
        for _, row in results.iterrows():
            values: list[str] = []
            for column in columns:
                value = row.get(column, np.nan)
                if pd.isna(value):
                    values.append("—")
                elif column.endswith("_pct"):
                    values.append(f"{float(value):.1f}")
                elif column in {"factor_label", "parameter"}:
                    values.append(str(value))
                elif column == "ekf_samples":
                    values.append(str(int(value)))
                else:
                    try:
                        values.append(f"{float(value):.3f}")
                    except (TypeError, ValueError):
                        values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.extend([
            "", "## Metric definitions", "",
            "- **Horizontal RMSE:** square root of the mean squared 2D North/East position-error magnitude.",
            "- **Horizontal MAE:** mean 2D horizontal position-error magnitude (not component-wise MAE).",
            "- **P95 / max:** 95th percentile and maximum 2D horizontal error magnitude.",
            "- **vs baseline:** `(baseline EKF RMSE - run EKF RMSE) / baseline EKF RMSE * 100`.",
            "- **vs GPS:** `(run GPS RMSE - run EKF RMSE) / run GPS RMSE * 100`.",
            "- Only horizontal error is summarized because the current SIMSTATE CSV parser provides latitude/longitude, not an independent vertical ground-truth field.",
            "",
        ])
        output_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _generate_comparison_plots(
        results: pd.DataFrame,
        error_arrays: dict[str, np.ndarray],
        output_directory: Path,
        baseline_rmse: float,
    ) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        non_baseline = results.loc[~results["is_baseline"]].copy()

        # Overall improvement ranking.
        if not non_baseline.empty:
            ranked = non_baseline.dropna(subset=["improvement_vs_baseline_pct"]).sort_values(
                "improvement_vs_baseline_pct",
            )
            if not ranked.empty:
                fig_height = max(4.5, 0.38 * len(ranked) + 1.5)
                fig, axis = plt.subplots(figsize=(11, fig_height))
                labels = [f"{row.parameter} · {row.factor_label}" for row in ranked.itertuples()]
                values = ranked["improvement_vs_baseline_pct"].to_numpy(dtype=float)
                axis.barh(labels, values)
                axis.axvline(0.0, linewidth=1, linestyle="--")
                axis.set_xlabel("Horizontal RMSE improvement vs baseline (%)")
                axis.set_title("EKF parameter experiments ranked by improvement")
                axis.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                fig.savefig(output_directory / "improvement_vs_baseline.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        # Distribution across all runs. Boxplots expose spread and outliers.
        labels: list[str] = []
        arrays: list[np.ndarray] = []
        for row in results.itertuples():
            values = error_arrays.get(row.experiment)
            if values is None:
                continue
            values = values[np.isfinite(values)]
            if values.size:
                labels.append("Baseline" if row.is_baseline else f"{row.parameter}\n{row.factor_label}")
                arrays.append(values)
        if arrays:
            fig_height = max(5.0, 0.45 * len(labels) + 2.0)
            fig, axis = plt.subplots(figsize=(max(9.0, 1.1 * len(labels)), fig_height))
            try:
                axis.boxplot(arrays, tick_labels=labels, showfliers=True)
            except TypeError:  # Matplotlib < 3.9 used the `labels` keyword.
                axis.boxplot(arrays, labels=labels, showfliers=True)
            axis.set_ylabel("2D horizontal EKF error (m)")
            axis.set_title("EKF horizontal error distributions by experiment")
            axis.grid(True, axis="y", alpha=0.3)
            axis.tick_params(axis="x", labelrotation=35)
            fig.tight_layout()
            fig.savefig(output_directory / "horizontal_error_boxplots.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # One sensitivity plot per parameter; factor axes are not mixed across parameters.
        for parameter, group in non_baseline.groupby("parameter", sort=True):
            numeric_group = group.dropna(subset=["factor", "ekf_horizontal_rmse_m"]).sort_values("factor")
            if numeric_group.empty:
                continue
            fig, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                numeric_group["factor"], numeric_group["ekf_horizontal_rmse_m"],
                marker="o", label="EKF horizontal RMSE",
            )
            axis.axhline(baseline_rmse, linestyle="--", label="Baseline EKF RMSE")
            if "measurement_horizontal_rmse_m" in numeric_group.columns:
                axis.plot(
                    numeric_group["factor"], numeric_group["measurement_horizontal_rmse_m"],
                    marker="s", linestyle=":", label="GPS measurement RMSE",
                )
            axis.set_xlabel("Parameter factor relative to baseline")
            axis.set_ylabel("Horizontal RMSE (m)")
            axis.set_title(f"EKF sensitivity: {parameter}")
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            filename = f"rmse_vs_factor_{BatchExperimentAnalyzer._safe_filename(str(parameter))}.png"
            fig.savefig(output_directory / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def run(self, root_directory: str | Path, generate_individual_plots: bool = True) -> pd.DataFrame:
        root = Path(root_directory).expanduser().resolve()
        experiments = self.discover_experiments(root)
        baseline_spec = next(experiment for experiment in experiments if experiment.is_baseline)

        rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        error_arrays: dict[str, np.ndarray] = {}
        baseline_row: dict[str, Any] | None = None
        print(f"Found {len(experiments) - 1} parameter-factor experiment(s).")
        for experiment in experiments:
            print(f"\nAnalyzing: {experiment.name}")
            try:
                row, errors = self._analyze_one(experiment, generate_plots=generate_individual_plots)
                rows.append(row)
                error_arrays[experiment.name] = errors
                if experiment.is_baseline:
                    baseline_row = row
            except Exception as error:
                message = f"FAILED: {experiment.directory}: {error}"
                if experiment.is_baseline:
                    raise RuntimeError(f"Could not analyze baseline. {message}") from error
                failures.append({
                    "experiment": experiment.name,
                    "directory": str(experiment.directory),
                    "error": str(error),
                })
                print(message, file=sys.stderr)

        if baseline_row is None:
            raise RuntimeError("Baseline results are missing; comparison cannot be performed")

        baseline_rmse = float(baseline_row["ekf_horizontal_rmse_m"])
        if not np.isfinite(baseline_rmse):
            raise RuntimeError(
                "Baseline EKF horizontal RMSE could not be calculated. Check that "
                "position measurements, EKF data, and SIMSTATE have overlapping timestamps."
            )
        for row in rows:
            run_ekf_rmse = float(row["ekf_horizontal_rmse_m"])
            run_gps_rmse = float(row["measurement_horizontal_rmse_m"])
            row["improvement_vs_baseline_pct"] = (
                (baseline_rmse - run_ekf_rmse) / baseline_rmse * 100.0
                if np.isfinite(baseline_rmse) and baseline_rmse > 0 and np.isfinite(run_ekf_rmse)
                else np.nan
            )
            row["improvement_vs_gps_pct"] = (
                (run_gps_rmse - run_ekf_rmse) / run_gps_rmse * 100.0
                if np.isfinite(run_gps_rmse) and run_gps_rmse > 0 and np.isfinite(run_ekf_rmse)
                else np.nan
            )

        results = pd.DataFrame(rows)
        results = results.sort_values(
            by=["is_baseline", "parameter", "factor"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

        comparison_csv = root / "comparison.csv"
        comparison_markdown = root / "comparison.md"
        results.to_csv(comparison_csv, index=False, float_format="%.8g")
        self._write_markdown_summary(results, comparison_markdown)
        skipped_csv = root / "skipped_experiments.csv"
        if failures:
            pd.DataFrame(failures).to_csv(skipped_csv, index=False)
        elif skipped_csv.exists():
            skipped_csv.unlink()
        comparison_plot_directory = root / "comparison_plots"
        self._generate_comparison_plots(
            results, error_arrays, comparison_plot_directory, baseline_rmse,
        )

        print("\nBatch analysis complete.")
        print(f"  Summary CSV:      {comparison_csv}")
        print(f"  Summary Markdown: {comparison_markdown}")
        print(f"  Comparison plots: {comparison_plot_directory}")
        if failures:
            print(f"  Skipped experiments: {skipped_csv} ({len(failures)} failed)")
        return results


def _default_csv_filenames() -> tuple[str, str, str]:
    """Obtain the project's existing logger output filenames."""
    position_config = PositionMeasurementLoggerConfig()
    logger_config = SitlLoggerConfig()
    return (
        Path(position_config.output_path).name,
        Path(logger_config.ekf_output_path).name,
        Path(logger_config.simstate_output_path).name,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot one SITL run or batch-compare EKF parameter experiments.",
    )
    parser.add_argument("--root", help="Batch experiment root containing baseline/ and parameter/factor/ folders.")
    parser.add_argument("--baseline-dir", default="baseline", help="Baseline folder name inside --root (default: baseline).")
    parser.add_argument("--position-measurement", metavar="CSV", help="Single-run position measurement CSV.")
    parser.add_argument("--sitl-ekf-state", metavar="CSV", help="Single-run EKF state CSV.")
    parser.add_argument("--sitl-simstate", metavar="CSV", help="Single-run SIMSTATE CSV.")
    parser.add_argument("--position-filename", help="CSV filename inside each batch experiment folder; defaults to logger config.")
    parser.add_argument("--ekf-filename", help="CSV filename inside each batch experiment folder; defaults to logger config.")
    parser.add_argument("--simstate-filename", help="CSV filename inside each batch experiment folder; defaults to logger config.")
    parser.add_argument("--no-individual-plots", action="store_true", help="Skip per-experiment plots; still calculate tables and comparison plots.")
    args = parser.parse_args(argv)

    plot_config = SitlEkfPlotterConfig()
    geographic_config = GeographicTransformConfig()

    if args.root:
        default_position, default_ekf, default_simstate = _default_csv_filenames()
        analyzer = BatchExperimentAnalyzer(
            plot_config=plot_config,
            geographic_config=geographic_config,
            position_filename=args.position_filename or default_position,
            ekf_filename=args.ekf_filename or default_ekf,
            simstate_filename=args.simstate_filename or default_simstate,
            baseline_directory_name=args.baseline_dir,
        )
        analyzer.run(args.root, generate_individual_plots=not args.no_individual_plots)
        return

    # Original single-run mode remains available.
    if any((args.position_measurement, args.sitl_ekf_state, args.sitl_simstate)):
        position_path = args.position_measurement
        ekf_path = args.sitl_ekf_state
        simstate_path = args.sitl_simstate
    else:
        position_config = PositionMeasurementLoggerConfig()
        logger_config = SitlLoggerConfig()
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
        print("Skipping error plots: external measurements and SIMSTATE are required.")


if __name__ == "__main__":
    main()
