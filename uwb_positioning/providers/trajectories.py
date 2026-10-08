import math

from uwb_positioning.config import CircleConfig

def calculate_circle_position(elapsed_time: float,
                              speed: float,
                              config: CircleConfig,
                              ) -> tuple[float, float, float]:
    angular_speed = speed / config.radius
    angle = angular_speed * elapsed_time

    x = config.radius * math.cos(angle)
    y = config.radius * math.sin(angle)
    z = config.altitude

    if config.vertical_movement:
        z += 0.5 * math.sin(0.5 * angle)

    return x, y, z
