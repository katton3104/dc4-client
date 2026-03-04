import numpy as np
import json
from typing import List
from pprint import pprint

from schema import MappingData, GridData
from simulation_database import MapDBManager
from grid_database import GridDBManager

class GridMaker:
    def __init__(self):
        self.map_db_manager = MapDBManager()
        self.grid_db_manager = GridDBManager()
        # self.map: List[MappingData] = self.map_db_manager.get_data()
        self.cw_map: List[MappingData] = self.map_db_manager.get_cw_data()
        self.ccw_map: List[MappingData] = self.map_db_manager.get_ccw_data()

    def create_grid(self):
        # デジタルカーリングのプレイエリアが(-2.085, 32.004)~(2.085, 40.234)(m)であるため、0.1m間隔でグリッドを作成
        x_lower = -20
        x_upper = 20
        y_lower = 320
        y_upper = 402

        self.grid = []
        for i in range(x_lower, x_upper + 1, 1):
            for j in range(y_lower, y_upper + 1, 1):
                self.grid.append((i / 10, j / 10))

        print(f"Grid size: {len(self.grid)}")

    def save_grid_to_db(self):
        # 0.1m間隔のグリッドに一番近い位置のデータを探して、その際の速度・位置を保尊
        self.create_grid()
        for position in self.grid:
            data: GridData = None
            x, y = position
            print(f"Processing position: {x} {y}")
            cw_closest_distance = 100 # 適当な値で初期化
            ccw_closest_distance = 100
            closest_distance = 100
            # 近い位置のデータを探す
            cw_approximate_value_id = 0
            ccw_approximate_value_id = 0
            approximate_value_id = 0
            for i in range(len(self.cw_map)):
                distance = np.sqrt(
                    (self.cw_map[i].position_x - x) ** 2 + (self.cw_map[i].position_y - y) ** 2
                )
                if distance < cw_closest_distance:
                    cw_closest_distance = distance
                    cw_approximate_value_id = i
            for i in range(len(self.ccw_map)):
                distance = np.sqrt(
                    (self.ccw_map[i].position_x - x) ** 2 + (self.ccw_map[i].position_y - y) ** 2
                )
                if distance < ccw_closest_distance:
                    ccw_closest_distance = distance
                    ccw_approximate_value_id = i

                #     ccw_approximate_value_id = i
            # if cw_closest_distance < 0.03 and ccw_closest_distance < 0.03:
            if cw_closest_distance < 0.03 and ccw_closest_distance < 0.03:
                data: GridData = GridData(
                    position_x=x,
                    position_y=y,
                    cw_velocity_x=self.cw_map[cw_approximate_value_id].velocity_x,
                    cw_velocity_y=self.cw_map[cw_approximate_value_id].velocity_y,
                    cw_angular_velocity=self.cw_map[cw_approximate_value_id].angular_velocity * (-np.pi/2),
                    ccw_velocity_x=self.ccw_map[ccw_approximate_value_id].velocity_x,
                    ccw_velocity_y=self.ccw_map[ccw_approximate_value_id].velocity_y,
                    ccw_angular_velocity=self.ccw_map[ccw_approximate_value_id].angular_velocity * (-np.pi/2),
                )
            elif cw_closest_distance < 0.03:
                data: GridData = GridData(
                    position_x=x,
                    position_y=y,
                    cw_velocity_x=self.cw_map[cw_approximate_value_id].velocity_x,
                    cw_velocity_y=self.cw_map[cw_approximate_value_id].velocity_y,
                    cw_angular_velocity=self.cw_map[cw_approximate_value_id].angular_velocity * (-np.pi/2),
                    ccw_velocity_x=None,
                    ccw_velocity_y=None,
                    ccw_angular_velocity=None,
                )
            elif ccw_closest_distance < 0.03:
                data: GridData = GridData(
                    position_x=x,
                    position_y=y,
                    cw_velocity_x=None,
                    cw_velocity_y=None,
                    cw_angular_velocity=None,
                    ccw_velocity_x=self.ccw_map[ccw_approximate_value_id].velocity_x,
                    ccw_velocity_y=self.ccw_map[ccw_approximate_value_id].velocity_y,
                    ccw_angular_velocity=self.ccw_map[ccw_approximate_value_id].angular_velocity * (-np.pi/2),
                )

            if data is not None:
                self.grid_db_manager.add_data(data)
            else:
                raise ValueError(
                    f"Closest distance is too large: {closest_distance} for position: {position}"
                )
        print("Grid data saved to database.")

    def update_grid_data(self):
        self.create_grid()

        for position in self.grid:
            x, y = position
            closest_distance = 100
            approximate_value_id = 0
            for i in range(len(self.map)):
                distance = np.sqrt(
                    (self.map[i].position_x - x) ** 2 + (self.map[i].position_y - y) ** 2
                )
                if distance < closest_distance:
                    closest_distance = distance
                    approximate_value_id = i
            if closest_distance < 0.03:
                print(f"Updating position: {x} {y} with closest distance: {closest_distance}")
                self.grid_db_manager.update_cw_velocity(
                    position_x=x,
                    position_y=y,
                    cw_velocity_x=self.map[approximate_value_id].velocity_x,
                    cw_velocity_y=self.map[approximate_value_id].velocity_y,
                    cw_angular_velocity=self.map[approximate_value_id].angular_velocity * (-np.pi/2),
                )
            # else:
            #     raise ValueError(
            #         f"Closest distance is too large: {closest_distance} for position: {position}"
            #     )


if __name__ == "__main__":
    grid_maker = GridMaker()
    grid_maker.save_grid_to_db()
    # grid_maker.update_grid_data()
