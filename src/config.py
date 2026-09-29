from dataclasses import dataclass


@dataclass
class TransformConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0

    local_x_heading_deg: float = 0.0

@dataclass
class SimulatorConfig:
    radius: float = 2.0
    altitude: float = 1.0
    angular_speed: float = 0.3
    noise_std: float = 0.02

@dataclass
class AppConfig:
    transform: TransformConfig
    simulator: SimulatorConfig

def load_config() -> AppConfig:
    return AppConfig(
            transform=TransformConfig(),
            simulator=SimulatorConfig(),
            )
