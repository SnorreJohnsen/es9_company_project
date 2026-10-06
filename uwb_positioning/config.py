from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    radius: float = 5.0
    altitude: float = 1.0
    angular_speed: float = 0.3
    noise_std: float = 0.05
    vertical_movement: bool = False

@dataclass
class TransformConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0

    local_x_heading_deg: float = 0.0

@dataclass
class ValidationConfig:
    x_min: float = -10.0
    x_max: float = 10.0

    y_min: float = -10.0
    y_max: float = 10.0

    z_min: float = -5.0
    z_max: float = 5.0

    maximum_velocity: float = 10.0

@dataclass
class MavlinkConfig:
    connection_string: str = "udpout:127.0.0.1:14550"
    source_system: int = 1
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
