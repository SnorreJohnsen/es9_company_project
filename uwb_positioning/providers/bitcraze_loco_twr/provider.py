import re
import time

import serial

from uwb_positioning.config import BitcrazeLocoTwrProviderConfig
from uwb_positioning.models import RangeMeasurement, RangeSetMeasurement
from uwb_positioning.providers.base import PositionProvider

# ^                       Start of line
# \s*                     Optional whitespace
# distance                Match literal "distance"
# \s+                     One or more whitespace characters
# (?P<anchor_id>\d+)      Capture anchor ID (digits)
# \s*:\s*                 Match colon with optional whitespace
# (?P<distance_mm>\d+)    Capture distance (digits)
# \s*mm                   Match unit "mm" with optional whitespace
# \s*                     Optional trailing whitespace
# $                       End of line
# re.IGNORECASE           Ignore uppercase/lowercase differences_
_DISTANCE_LINE = re.compile(r"^\s*distance\s+(?P<anchor_id>\d+)\s*:\s*"
                            r"(?P<distance_mm>\d+)\s*mm\s*$",
                            re.IGNORECASE,
                            )

class BitcrazeLocoTwrProvider(PositionProvider):
    def __init__(self, config: BitcrazeLocoTwrProviderConfig) -> None:
        self.config = config
        self._serial = None
        self._pending: dict[int, RangeMeasurement] = {}
        self._anchor_ids = set(config.anchor_ids)

    def connect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        
        self._serial = serial.Serial(port=self.config.port,
                                     baudrate=self.config.baudrate,
                                     timeout=self.config.serial_timeout_s,
                                     )
        self._serial.reset_input_buffer()
        self._pending.clear()

    def get_measurement(self,
                        timeout_s: float | None = 1.0,
                        ) -> RangeSetMeasurement | None:
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Bitcraze Loco TWR provider connected")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        max_span_us = int(self.config.max_range_set_span_s * 1_000_000)

        while deadline is None or time.monotonic() < deadline:
            raw_line = self._serial.readline()
            if not raw_line:
                continue

            line = raw_line.decode("ascii", errors="ignore").strip()
            sample = self.parse_distance_line(line)
            if sample is None or sample.anchor_id not in self._anchor_ids:
                continue

            self._pending[sample.anchor_id] = sample
            if not self._anchor_ids.issubset(self._pending):
                continue
            
            samples = self._pending.copy()
            timestamps = [item.timestamp_us for item in samples.values()]
            first_us = min(timestamps)
            last_us = max(timestamps)

            if last_us - first_us <= max_span_us:
                self._pending.clear()
                return RangeSetMeasurement(timestamp_us=(first_us + last_us) // 2,
                                           ranges=samples,
                                           )
            # discard old mesurements, keep recent ones while collecting a new and temporally consitent set
            cutoff_us = last_us - max_span_us
            self._pending = {anchor_id: sample
                             for anchor_id, sample in self._pending.items()
                             if sample.timestamp_us >= cutoff_us
                             }

            return None


    @staticmethod
    def parse_distance_line(line: str,
                            *,
                            timestamp_us: int | None = None,
                            ) -> RangeMeasurement | None:
        match = _DISTANCE_LINE.match(line)
        if match is None:
            return None

        anchor_id = int(match.group("anchor_id"))
        distance_m = int(match.group("distance_mm")) / 1000.0
        if distance_m <= 0:
            return None

        if timestamp_us is None:
            timestamp_us = time.monotonic_ns() // 1000

        return RangeMeasurement(timestamp_us=timestamp_us,
                                anchor_id=anchor_id,
                                distance_m=distance_m,
                                )
    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._pending.clear()
