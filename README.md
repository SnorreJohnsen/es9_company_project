# ES9 Company Project - Ultra-Wideband (UWB) Real Time Location System for Indoor Positioning of Drone

## Dependencies

python packages
- `matplotlib`
- `pytest`
- `pymavlink`
- `numpy`
- `pandas`


## Usage

Run the desired application from the repository root.

```bash
python -m uwb_positioning.<main>
```

Replace `<main>` with:

- `simulator_main` - Simulates input and output to MAVLink.
- `ardupilot_sitl_gps_input_main` - Uses the ArduPilot SITL GPS simulated positions and outputs to MAVLink as ODOMETRY message.
- `ardupilot_sitl_odometry_main` - Uses the ArduPilot SITL GPS simulated positions and outputs to MAVLink as GPS_INPUT message.

Edit `uwb_positioning/config.py` to change the application configuration.

### ArduPilot SITL

**_NOTE:_** This has only been used with WSL2 running Ubuntu 24.04.

Start ArduPilot SITL in a terminal:

```bash
cd path/to/ardupilot/ArduCopter
```

For the first time run:

```bash
sim_vehicle.py -v ArduCopter -f quad --map --console -w
```

`-w` is to reset the simulation environment with default parameters.

Load parameters.

```text
param load path/to/paramters.param
```

Reboot may be required after loading parameters.

For subsequent runs use the following:

```bash
sim_vehicle.py -v ArduCopter -f quad --map --console \
--out=udp:127.0.0.1:14550 \
--out=udp:127.0.0.1:14551 \
--out=udp:127.0.0.1:14552
```

These UDP ports should match the ports in the `uwb_positioning/config.py`.
The purpose of the output ports is:

- `--out=udp:127.0.0.1:14550` the position measurement which is transmitted to the drone.
- `--out=udp:127.0.0.1:14551` the SITL provider to simulate real UWB position data.
- `--out=udp:127.0.0.1:14552` SITL logger which logs LOCAL_POSITION_NED and SIMSTATE MAVLink messages for EKF analysis.

When the simulation environment has started run the following command in the MAVProxy terminal.

```text
set streamrate -1
```

This is to prevent ArduPilot SITL from reverting the requested message rate to default values.


In a second terminal run:

```bash
cd path/to/es9_company_project

python -m uwb_positioning.ardupilot_sitl_<output_type>_main
```

In the MAVProxy console set the vehicle to GUIDED mode, arm it and take off.

```text
mode guided
arm throttle
takeoff <int>
```

Speed [m/s] can be changed with:

```text
param set WP_SPD <int | float>
```

Move to destination by right-click the map and choose `fly to`


For automated flight run the following commands. 

```text
wp load <flypath.txt>
mode auto
```

## Tests 

Run the automated tests from the repository root:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

