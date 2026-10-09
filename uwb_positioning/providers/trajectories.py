import math

from uwb_positioning.config import CircleConfig, LineConfig

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

def calculate_line_position(elapsed_time: float,
                            speed: float,
                            config: LineConfig,
                            ) -> tuple[float, float, float]:
    difference_x = config.target_x - config.start_x
    difference_y = config.target_y - config.start_y
    difference_z = config.target_z - config.start_z

    path_length = math.sqrt(difference_x**2
                            + difference_y**2
                            + difference_z**2
                            )
    if path_length == 0.0:
        return(config.start_x,
               config.start_y,
               config.start_z,
               )

    distance_travelled = speed * elapsed_time
    
    if config.repeat:
        progress = _calculate_repeating_progress(distance_travelled,
                                                path_length,
                                                )
    else:
        progress = min(distance_travelled / path_length,
                       1.0,
                       )
    x = config.start_x + progress * difference_x   
    y = config.start_y + progress * difference_y   
    z = config.start_z + progress * difference_z   
    
    return x, y, z

def _calculate_repeating_progress(distance_travelled: float,
                                  path_length: float,
                                  ) -> float:
    cycle_length = 2.0 * path_length
    distance_in_cycle = distance_travelled % cycle_length

    if distance_in_cycle <= path_length:
        return distance_in_cycle / path_length

    return (cycle_length - distance_in_cycle) / path_length
