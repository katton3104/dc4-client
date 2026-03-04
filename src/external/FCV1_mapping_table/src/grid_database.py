import pathlib
import numpy as np

from sqlalchemy import create_engine, Float, Integer, Column, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from schema import GridData

file_path = pathlib.Path(__file__).parents[0]
file_path /= "grid.sqlite3"
sqlite_url = f"sqlite:///{file_path}"

engine = create_engine(sqlite_url, echo=False)
Session = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Grid(Base):
    # 実際に使用して頂くデータベースのテーブルです
    # このテーブル名とこの行の名前が一番重要なデータです
    __tablename__ = "sheet_grid"
    id = Column(Integer, primary_key=True)
    position_x = Column(Float)
    position_y = Column(Float)
    cw_velocity_x = Column(Float)
    cw_velocity_y = Column(Float)
    cw_angular_velocity = Column(Float)
    ccw_velocity_x = Column(Float)
    ccw_velocity_y = Column(Float)
    ccw_angular_velocity = Column(Float)


class GridDBManager:
    def __init__(self):
        self.create_table()

    def create_table(self):
        """テーブルを作成する"""
        try:
            with engine.begin() as conn:
                Base.metadata.create_all(conn)
        except Exception as e:
            print(f"Error creating table: {e}")

    def add_data(self, data: GridData):
        """データを追加する
        :param data: GridData
        """
        try:
            with Session() as session:
                new_data = Grid(
                    position_x=data.position_x,
                    position_y=data.position_y,
                    cw_velocity_x=data.cw_velocity_x,
                    cw_velocity_y=data.cw_velocity_y,
                    cw_angular_velocity=data.cw_angular_velocity,
                    ccw_velocity_x=data.ccw_velocity_x,
                    ccw_velocity_y=data.ccw_velocity_y,
                    ccw_angular_velocity=data.ccw_angular_velocity,
                )
                session.add(new_data)
                session.commit()
        except Exception as e:
            print(f"Error adding data: {e}")

    def get_data(self) -> list[GridData]:
        """全てのデータを取得する
        :return: list[GridData]
        """
        try:
            with Session() as session:
                stmt = select(Grid)
                result = session.execute(stmt).scalars().all()
                if result is None:
                    print("No data found in the database.")
                    return None
                all_grid_data = [
                    GridData(
                        position_x=row.position_x,
                        position_y=row.position_y,
                        velocity_x=row.cw_velocity_x,
                        velocity_y=row.cw_velocity_y,
                        angular_velocity=row.cw_angular_velocity,
                        ccw_velocity_x=row.ccw_velocity_x,
                        ccw_velocity_y=row.ccw_velocity_y,
                        ccw_angular_velocity=row.ccw_angular_velocity,
                    )
                    for row in result
                ]
                return all_grid_data
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
        
    def get_velocity(self, position_x, position_y) -> GridData:
        """指定した位置の速度を取得する
        :param position_x: x座標
        :param position_y: y座標
        :return: GridData
        """
        try:
            with Session() as session:
                stmt = (
                    select(Grid)
                    .where(Grid.position_x == position_x)
                    .where(Grid.position_y == position_y)
                )
                result = session.execute(stmt).scalars().first()
                if result is None:
                    print("This position does not exist in the database.")
                    return None
                return GridData(
                    position_x=result.position_x,
                    position_y=result.position_y,
                    cw_velocity_x=result.cw_velocity_x,
                    cw_velocity_y=result.cw_velocity_y,
                    cw_angular_velocity=result.cw_angular_velocity,
                    ccw_velocity_x=result.ccw_velocity_x,
                    ccw_velocity_y=result.ccw_velocity_y,
                    ccw_angular_velocity=result.ccw_angular_velocity,
                )
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None


if __name__ == "__main__":
    grid_db_manager = GridDBManager()
    x_positions = 0.0
    y_positions = 38.0
    velocity = grid_db_manager.get_velocity(position_x=x_positions, position_y=y_positions)
    print(velocity)

