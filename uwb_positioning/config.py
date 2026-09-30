from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    radius: float = 2.0
    altitude: float = 1.0
    angular_speed: float = 0.3
    noise_std: float = 0.02

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
class AppConfig:
    transform: TransformConfig
    validator: ValidationConfig
    simulator: SimulatorConfig

def load_config() -> AppConfig:
    return AppConfig(
            transform=TransformConfig(),
            validator=ValidationConfig(),
            simulator=SimulatorConfig(),
            )
