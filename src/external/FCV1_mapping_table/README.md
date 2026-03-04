# FCV1 座標・速度対応表
カーリングシートのプレイエリア(-2.085, 32.004) ~ (2.085, 40.234)内を0.1mおきにグリッドを作成して、その際の初速度・角速度を取得できるデータベースを作成しました。

## ライブラリインストール
```bash
pip install -r requirements.txt
```

## データの中身
```Python
class Grid(Base):
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
```
データとしては、デジタルカーリングにおけるプレーエリア内の座標(0.1mおき)とその座標に到達するために必要な初速度・角速度を用意してあります。

各positionは上記で説明した、0.1mおきのグリッドの座標が格納されています。 
cw(Clock Wise)はインターン、ccw(Counter ClockWise)はアウトターンという対応関係です。 
上記の変数において、頭にcwと付いている変数はインターンでの投球時にposition_x、position_yの一番近くに停止した際の投球情報です。(ccwはアウトターン用であり、それ以外は同様です。) angular_velocityには、fcv1のデフォルト値であるπ/2(または-π/2)に10倍した値が格納されています。

## 使用方法
**grid_database.py**を起動して頂けると、すぐに座標と速度の情報が出力されるかと思います。
**GridDBManager**クラス内の**get_velocity**関数を起動する際の引数として
- position_x
- position_y
の二つを用意して頂きます。
そちらを実行すると、
```
position_x=0.0 position_y=38.0 cw_velocity_x=-0.131 cw_velocity_y=2.388 cw_angular_velocity=-15.707963267948966 ccw_velocity_x=0.131 ccw_velocity_y=2.388 ccw_angular_velocity=15.707963267948966
```
といった形で出力されるかと思います。なお、カーリングシートの端を狙う場合は、cwまたはccwのどちらかはNoneが入っていることもあるので、そちらはご確認ください。