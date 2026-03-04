from pydantic import BaseModel
from typing import Optional


class GridData(BaseModel):
    position_x: float
    position_y: float
    cw_velocity_x: Optional[float] = None
    cw_velocity_y: Optional[float] = None
    cw_angular_velocity: Optional[float] = None
    ccw_velocity_x: Optional[float] = None
    ccw_velocity_y: Optional[float] = None
    ccw_angular_velocity: Optional[float] = None
