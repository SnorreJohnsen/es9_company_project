# ES9 Company Project - Ultra-Wideband (UWB) Real Time Location System for Indoor Positioning of Drone

## Dependencies

python packages
- `matplotlib`
- `pytest`
- `pymavlink`


## Usage

Run the desired application from the repository root.

```bash
python -m uwb_positioning.<main>
```

Replace `<main>` with:

- `simulator_main` - Simulates input and output to MAVLink.
- `ardupilot_sitl_main` - Uses the ArduPilot SITL GPS simulated positions and outputs to MAVLink.

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

param load path/to/es9_company_project/uwb_positioning/providers/ardupilot_sitl/external_positioning.parm
```

For subsequent runs use the following:

```bash
sim_vehicle.py -v ArduCopter -f quad --map --console \
--out=udp:127.0.0.1:14550 \
--out=udp:125.0.0.1:14551
```

These UDP ports should match the ports in the `uwb_positioning/config.py`.

In a second terminal run:

```bash
cd path/to/es9_company_project

python -m uwb_positioning.ardupilot_sitl_main
```

In the MAVProxy console set the vehicle to GUIDED mode, arm it and take off.

```text
mode guided
arm throttle
takeoff [int]
```

Speed [m/s] can be changed with:

```text
param set WP_SPD [int | float]
```

Move to destination by right-click the map and choose `fly to`

## Tests 

Run the automated tests from the repository root:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

