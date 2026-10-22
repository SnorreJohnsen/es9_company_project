from dataclasses import dataclass, field


# Simulator
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
                                         LineConfig # change here for trajectory
                                         )

# Ardupilot SITL
@dataclass
class SitlProviderConfig:
    connection_string: str = "udpin:127.0.0.1:14551"

@dataclass
class SitlLoggerConfig:
    connection_string: str = "udpin:127.0.0.1:14552"
    message_rate_hz: float = 20.0

    ekf_output_path: str = "uwb_positioning/csv_logging/logs/sitl_ekf_position.csv"
    simstate_output_path: str = "uwb_positioning/csv_logging/logs/sitl_simstate_position.csv"
    flush_every_row: bool = False

@dataclass
class SitlSimulationConfig:
    duration_s: float | None = None # None: stop simulation with ctrl+C

    horizontal_noise_std_m: float = 0.05
    vertical_noise_std_m: float = 0.05
    random_seed: int | None = 42

    delay_s: float = 0.1

# shared
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
    connection_string: str = "udpin:127.0.0.1:14550"
    source_system: int = 200
    source_component: int = 191 # MAV_COMP_ID_ONBOARD_COMPUTER

    send_rate_hz: float = 10.0
    enabled: bool = True

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
    transform: LocalTransformConfig
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
                              transform=LocalTransformConfig(), 
                              validator=ValidationConfig(), 
                              mavlink=MavlinkConfig(),
                              )
