from dataclasses import dataclass, field

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

@dataclass
class TransformConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0

    local_x_heading_deg: float = 0.0

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
    source_component: int = 191

    send_rate_hz: float = 10.0
    enabled: bool = False

@dataclass
class AppConfig:
    transform: TransformConfig
    validator: ValidationConfig
    mavlink: MavlinkConfig
    simulator: SimulatorConfig

def load_config() -> AppConfig:
    return AppConfig(
            transform=TransformConfig(),
            validator=ValidationConfig(),
            mavlink=MavlinkConfig(),
            simulator=SimulatorConfig(),
            )
