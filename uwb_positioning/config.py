from dataclasses import dataclass, field


############################# Simulator ####################################
@dataclass
class CircleConfig:
    radius: float = 5.0
    altitude: float = 1.0

    vertical_movement: bool = False

@dataclass
class LineConfig:
    start_x: float = 5.0 
    start_y: float = 5.0 
    start_z: float = 5.0 

    target_x: float = 11.0
    target_y: float = 11.0
    target_z: float = 11.0

    repeat: bool = False

TrajectoryConfig = CircleConfig | LineConfig # add | another config when new trajectory

@dataclass
class SimulatorConfig:
    speed_mps: float = 1.5
    noise_std: float = 0.05
    duration_s: float | None = None # None: stop simulation with ctrl+C
    random_seed: int | None = 42

    trajectory: TrajectoryConfig = field(default_factory=
                                         CircleConfig # change here for trajectory
                                         )

########################## Ardupilot SITL ##########################
@dataclass
class SitlProviderConfig:
    connection_string: str = "udpin:127.0.0.1:14551"
    message_rate_hz: float = 10.0

@dataclass
class SitlLoggerConfig:
    connection_string: str = "udpin:127.0.0.1:14552"
    message_rate_hz: float = 20.0

    ekf_output_path: str = "uwb_positioning/csv_logging/logs/sitl_ekf_position.csv"
    simstate_output_path: str = "uwb_positioning/csv_logging/logs/sitl_simstate_position.csv"
    flush_every_row: bool = False

@dataclass
class SitlSimulationConfig:
    duration_s: float | None = 130 # None: stop simulation with ctrl+C

    horizontal_noise_std_m: float = 0.05
    vertical_noise_std_m: float = 0.05
    random_seed: int | None = 42

    delay_s: float = 0.0

@dataclass
class SitlEkfPlotterConfig:
    lowpass_cutoff_hz: float = 0.1

    output_directory: str = "visualization/figures/sitl_ekf/"
    sync_tolerance_ms: float | None = 100.0
    include_invalid_measurements: bool = False
    
    figure_width: float = 12.0
    figure_height: float = 8.0
    dpi: int = 150
    histogram_bins: int | str = "auto"

######################### bitcraze twr loco #############################
@dataclass
class BitcrazeLocoTwrProviderConfig:
    port: str = "" # fx. "/dev/ttyACM0"
    baudrate: int = 115200 # not needed for USB CDC
    anchor_ids: tuple[int, ...] = tuple(range(8))
    serial_timeout_s: float = 0.10
    max_range_set_span_s: float = 0.50

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("Serial port required, fx. /dev/ttyACM0")
        if not self.anchor_ids:
            raise ValueError("At least one anchor id required")
        if len(set(self.anchor_ids)) != len(self.anchor_ids):
            raise ValueError("anchor_ids must be unique")
        if any(anchor_id < 0 or anchor_id > 7 for anchor_id in self.anchor_ids):
            raise ValueError("Anchor ids must be range 0 to 7")

@dataclass
class TrilaterationConfig:
    # anchors in FLU coordinates in meters
    anchors: dict[int, tuple[float, float, float]] = field(default_factory=lambda: {
        0: (0.0, 0.0, 0.0),
        7: (3.0, 0.0, 0.0),
        2: (3.0, 6.0, 0.0),
        5: (0.0, 6.0, 0.0),

        4: (0.0, 0.0, 3.0),
        3: (3.0, 0.0, 3.0),
        6: (3.0, 6.0, 3.0),
        1: (0.0, 6.0, 3.0),
        })
    minimum_anchors: int = 6
    max_nfev: int = 60

    loss: str = "soft_l1"
    loss_scale_m: float = 0.10 # residual scale in meters. controls when to reduce outlier influence

    max_residual_rms_m: float | None = None # max accepted rms distance error

######################### shared ########################################
@dataclass
class PositionMeasurementLoggerConfig:
    output_path: str = "uwb_positioning/csv_logging/logs/position_measurement.csv"
    flush_every_row: bool = False

@dataclass
class LocalTransformConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0

    local_x_heading_deg: float = 0.0

@dataclass
class GeographicTransformConfig:
    earth_radius_m: float = 6_378_137.0

    origin_latitude_deg: float = -353632621 / 1e7
    origin_longitude_deg: float = 1491652374 / 1e7 
    origin_altitude_m: float =   584090 / 1000

@dataclass
class ValidationConfig:
    x_min: float = -8.0
    x_max: float = 8.0

    y_min: float = -8.0
    y_max: float = 8.0

    z_min: float = -8.0
    z_max: float = 8.0

    maximum_velocity: float = 10.0

@dataclass
class MavlinkConfig:
    connection_string: str = "/dev/ttyACM0"
    baudrate: int = 115200
    source_system: int = 200
    source_component: int = 191 # MAV_COMP_ID_ONBOARD_COMPUTER

    send_rate_hz: float = 10.0
    enabled: bool = True


##################################### loaders #############################################
@dataclass
class ArdupilotSitlAppConfig:
    provider: SitlProviderConfig
    simulation: SitlSimulationConfig
    geographic_transformer: GeographicTransformConfig
    local_transformer: LocalTransformConfig
    validator: ValidationConfig
    mavlink: MavlinkConfig
    sitl_logger: SitlLoggerConfig
    position_measurement_logger: PositionMeasurementLoggerConfig

@dataclass
class SimulatorAppConfig:
    provider: SimulatorConfig
    geographic_transformer: GeographicTransformConfig
    local_transformer: LocalTransformConfig
    validator: ValidationConfig
    mavlink: MavlinkConfig

def load_ardupilot_sitl_config() -> ArdupilotSitlAppConfig:
    return ArdupilotSitlAppConfig(provider=SitlProviderConfig(),
                                  simulation=SitlSimulationConfig(),
                                  geographic_transformer=GeographicTransformConfig(),
                                  local_transformer=LocalTransformConfig(),
                                  validator=ValidationConfig(), 
                                  mavlink=MavlinkConfig(), 
                                  sitl_logger=SitlLoggerConfig(),
                                  position_measurement_logger=PositionMeasurementLoggerConfig(),
                                  )

def load_simulator_config() -> SimulatorAppConfig:
    return SimulatorAppConfig(provider=SimulatorConfig(), 
                              geographic_transformer=GeographicTransformConfig(),
                              local_transformer=LocalTransformConfig(),
                              validator=ValidationConfig(), 
                              mavlink=MavlinkConfig(),
                              )
