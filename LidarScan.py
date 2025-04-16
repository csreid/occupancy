from dataclasses import dataclass

@dataclass
class Range:
	angle: float
	distance: float

@dataclass
class LidarScan:
	ranges: list[Range]
