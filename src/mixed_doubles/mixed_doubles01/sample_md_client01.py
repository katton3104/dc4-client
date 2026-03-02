import asyncio
import json
import numpy as np
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FCV1_MAPPING_TABLE_PATH = PROJECT_ROOT / "external" / "FCV1_mapping_table"
FCV1_MAPPING_TABLE_SRC_PATH = FCV1_MAPPING_TABLE_PATH / "src"
if FCV1_MAPPING_TABLE_PATH.exists() and str(FCV1_MAPPING_TABLE_PATH) not in sys.path:
    sys.path.insert(0, str(FCV1_MAPPING_TABLE_PATH))
if FCV1_MAPPING_TABLE_SRC_PATH.exists() and str(FCV1_MAPPING_TABLE_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(FCV1_MAPPING_TABLE_SRC_PATH))

from load_secrets import username, password
from dc4client.dc_client import DCClient
from dc4client.send_data import TeamModel, MatchNameModel, PositionedStonesModel
from grid_database import GridDBManager

formatter = logging.Formatter(
    "%(asctime)s, %(name)s : %(levelname)s - %(message)s"
)

HOUSE_CENTER_X = 0.0
HOUSE_CENTER_Y = 38.405
HOUSE_RADIUS = 1.83
PLOT_X_MIN = -2.085
PLOT_X_MAX = 2.085
PLOT_Y_MIN = 32.004
PLOT_Y_MAX = 40.234
PLOT_WIDTH = 61
PLOT_HEIGHT = 25


def get_active_stones(state_data):
    stones = []
    if state_data.stone_coordinate is None or state_data.stone_coordinate.data is None:
        return stones

    for team_name, coords in state_data.stone_coordinate.data.items():
        for stone in coords:
            if stone is None:
                continue
            x = float(stone.x)
            y = float(stone.y)
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                continue
            dist = np.hypot(x - HOUSE_CENTER_X, y - HOUSE_CENTER_Y)
            stones.append({"team": team_name, "x": x, "y": y, "dist": dist})
    return stones


def get_stone_centroid(state_data):
    stones = get_active_stones(state_data)
    if not stones:
        return 0.0, 0.0, 0
    centroid_x = float(np.mean([stone["x"] for stone in stones]))
    centroid_y = float(np.mean([stone["y"] for stone in stones]))
    return centroid_x, centroid_y, len(stones)


def get_no1_stone(state_data):
    stones = get_active_stones(state_data)
    if not stones:
        return None
    in_house = [stone for stone in stones if stone["dist"] <= HOUSE_RADIUS]
    if not in_house:
        return None
    return min(in_house, key=lambda s: s["dist"])


def render_ascii_board(state_data, my_team_name):
    width = PLOT_WIDTH
    height = PLOT_HEIGHT
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    def to_col(x):
        ratio = (x - PLOT_X_MIN) / (PLOT_X_MAX - PLOT_X_MIN)
        return int(np.clip(round(ratio * (width - 1)), 0, width - 1))

    def to_row(y):
        ratio = (y - PLOT_Y_MIN) / (PLOT_Y_MAX - PLOT_Y_MIN)
        return int(np.clip(round((1.0 - ratio) * (height - 1)), 0, height - 1))

    canvas[to_row(HOUSE_CENTER_Y)][to_col(HOUSE_CENTER_X)] = "+"

    stones = get_active_stones(state_data)
    no1 = get_no1_stone(state_data)

    for stone in sorted(stones, key=lambda s: s["dist"], reverse=True):
        col = to_col(stone["x"])
        row = to_row(stone["y"])
        marker = "M" if stone["team"] == my_team_name else "E"
        canvas[row][col] = marker

    if no1 is not None:
        col = to_col(no1["x"])
        row = to_row(no1["y"])
        canvas[row][col] = "N" if no1["team"] == my_team_name else "C"

    lines = []
    header = (
        f"Board end={state_data.end_number} shot={state_data.total_shot_number} "
        f"next={state_data.next_shot_team} winner={state_data.winner_team}"
    ) + f" area[x:{PLOT_X_MIN:.3f}..{PLOT_X_MAX:.3f}, y:{PLOT_Y_MIN:.3f}..{PLOT_Y_MAX:.3f}]"
    lines.append(header)
    lines.append("+" + "-" * width + "+")
    for row in canvas:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append("Legend: N=No.1(my), C=No.1(enemy), M=my stone, E=enemy stone, +=tee")
    return "\n".join(lines)


def clamp_target_for_grid(x, y):
    x = round(float(np.clip(x, -2.0, 2.0)), 1)
    y = round(float(np.clip(y, 32.0, 40.2)), 1)
    return x, y


def shot_from_grid_velocity(grid_db_manager, target_x, target_y, use_cw):
    x, y = clamp_target_for_grid(target_x, target_y)
    grid_data = grid_db_manager.get_velocity(position_x=x, position_y=y)
    if grid_data is None:
        return None

    if use_cw:
        if (
            grid_data.cw_velocity_x is None
            or grid_data.cw_velocity_y is None
            or grid_data.cw_angular_velocity is None
        ):
            return None
        vx = grid_data.cw_velocity_x
        vy = grid_data.cw_velocity_y
        w = grid_data.cw_angular_velocity
    else:
        if (
            grid_data.ccw_velocity_x is None
            or grid_data.ccw_velocity_y is None
            or grid_data.ccw_angular_velocity is None
        ):
            return None
        vx = grid_data.ccw_velocity_x
        vy = grid_data.ccw_velocity_y
        w = grid_data.ccw_angular_velocity

    return np.hypot(vx, vy), np.arctan2(vy, vx), -w / 10.0, x, y, vx, vy, w

def build_scoreboard(team0_scores, team1_scores, end_count, my_team_name):
    headers = [f"E{i}" for i in range(1, end_count + 1)] + ["TOTAL"]
    team0_total = sum(team0_scores[:end_count])
    team1_total = sum(team1_scores[:end_count])
    team0_values = [str(v) for v in team0_scores[:end_count]] + [str(team0_total)]
    team1_values = [str(v) for v in team1_scores[:end_count]] + [str(team1_total)]

    team0_label = "team0 (YOU)" if my_team_name == "team0" else "team0"
    team1_label = "team1 (YOU)" if my_team_name == "team1" else "team1"

    score_width = max(5, *(len(v) for v in headers), *(len(v) for v in team0_values), *(len(v) for v in team1_values))
    team_col_width = max(len(team0_label), len(team1_label), 4)

    def make_row(label, values):
        return f"| {label:<{team_col_width}} | " + " | ".join(f"{v:>{score_width}}" for v in values) + " |"

    sep = "+-" + "-" * team_col_width + "-+-" + "-+-".join("-" * score_width for _ in headers) + "-+"
    lines = [
        f"Scoreboard (End {end_count})",
        sep,
        make_row("TEAM", headers),
        sep,
        make_row(team0_label, team0_values),
        make_row(team1_label, team1_values),
        sep,
    ]
    return "\n".join(lines)

def resolve_display_end_count(team0_scores, team1_scores, regulation_ends=8):
    available_ends = min(len(team0_scores), len(team1_scores))
    if available_ends <= regulation_ends:
        return available_ends

    cum0 = 0
    cum1 = 0
    for i in range(available_ends):
        cum0 += team0_scores[i]
        cum1 += team1_scores[i]
        if i + 1 < regulation_ends:
            continue
        if cum0 != cum1:
            return i + 1

    return available_ends

async def main():
    # 最初のエンドにおいて、team0が先攻、team1が後攻です。
    # デフォルトではteam1となっており、先攻に切り替えたい場合は下記を
    # team_name=MatchNameModel.team0
    # に変更してください
    json_path = Path(__file__).parents[1] / "match_id.json"

    # match_idの読み込みます。
    with open(json_path, "r") as f:
        match_id = json.load(f)
    # クライアントの初期化（ログレベルはデフォルトでINFO、保存機能はデフォルトでTrue）
    client = DCClient(match_id=match_id, username=username, password=password, match_team_name=MatchNameModel.team1, auto_save_log=True, log_dir="logs")

    # ここで、接続先のサーバのアドレスとポートを指定します。
    # デフォルトではlocalhost:5000となっています。
    # こちらは接続先に応じて変更してください。
    client.set_server_address(host="localhost", port=5001)

    # チーム設定の読み込み
    with open("md_team_config.json", "r") as f:
        data = json.load(f)
    client_data = TeamModel(**data)

    # ログ設定(不要であれば削除してください)
    logger = logging.getLogger("SampleMDClient")
    logger.setLevel(level=logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"client_data.team_name: {client_data.team_name}")
    logger.debug(f"client_data: {client_data}")


    # チーム情報をサーバに送信します。
    # 相手のクライアントも同様にチーム情報を送信するまで待機します。
    # 送信後、自チームの名前を受け取ります(team0 または team1)。
    # 両チームが揃うと試合が開始されます。
    # 最初の置き石を設定するチームが、サーバに置き石の設定を送信したら思考時間のカウントが始まります。
    # そのため、AIの初期化などはこの前に行ってください。
    match_team_name: MatchNameModel = await client.send_team_info(client_data)
    grid_db_manager = GridDBManager()
    first_end_number = None
    last_logged_final_end_total_shot = None
    regulation_ends = 8

    try:
        async for state_data in client.receive_state_data():
            if first_end_number is None and state_data.end_number is not None:
                first_end_number = state_data.end_number
            final_end_number = (
                regulation_ends - 1 if first_end_number == 0 else regulation_ends
            )
            if (
                state_data.end_number == final_end_number
                and state_data.total_shot_number is not None
                and state_data.total_shot_number != last_logged_final_end_total_shot
            ):
                logger.info("\n" + render_ascii_board(state_data, match_team_name.value))
                last_logged_final_end_total_shot = state_data.total_shot_number

            # ゲーム終了の判定
            if (winner_team := client.get_winner_team()) is not None:
                logger.info("\n" + render_ascii_board(state_data, match_team_name.value))
                if state_data.score is not None:
                    team0_scores = state_data.score.team0
                    team1_scores = state_data.score.team1
                    display_end_count = resolve_display_end_count(team0_scores, team1_scores, regulation_ends=8)
                    if display_end_count > 0:
                        scoreboard = build_scoreboard(
                            team0_scores,
                            team1_scores,
                            display_end_count,
                            match_team_name.value,
                        )
                        logger.info(f"\n{scoreboard}")
                if winner_team == match_team_name.value:
                    logger.info(f"You won! Winner: {winner_team}")
                else:
                    logger.info(f"You lost. Winner: {winner_team}")
                break

            next_shot_team = client.get_next_team()

            # AIを実装する際の処理はこちらになります。
            # 最初の置き石を設定するチームの場合、最初の状態データ受信時に置き石の情報を送信します。
            if state_data.next_shot_team is None and state_data.mix_doubles_settings is not None and state_data.last_move is None:
                if state_data.mix_doubles_settings.end_setup_team == match_team_name:
                    logger.info("You select the positioned stones.")
                    # 置き石のパターンを選択します。
                    # 以下のいずれかを選択してください。
                    # PositionedStonesModel.center_guard -> 現エンド: ガードを中央に置き、先攻
                    # PositionedStonesModel.center_house -> 現エンド: ハウスを中央に置き、後攻
                    # PositionedStonesModel.pp_left      -> 現エンド: パワープレイを実施し、左側に置き、後攻
                    # PositionedStonesModel.pp_right     -> 現エンド: パワープレイを実施し、右側に置き、後攻
                    positioned_stones = PositionedStonesModel.pp_left

                    # 置き石の情報をサーバに送信します。
                    await client.send_positioned_stones_info(positioned_stones)

            if next_shot_team == match_team_name:
                total_shots_played = state_data.total_shot_number or 0
                next_shot_index = total_shots_played + 1
                centroid_x, centroid_y, stone_count = get_stone_centroid(state_data)
                use_cw = centroid_x >= 0.0
                rotation_label = "cw" if use_cw else "ccw"
                logger.info(
                    "Stone centroid before shot: count=%d, x=%.3f, y=%.3f -> rotation=%s",
                    stone_count,
                    centroid_x,
                    centroid_y,
                    rotation_label,
                )

                target_x, target_y = HOUSE_CENTER_X, HOUSE_CENTER_Y
                shot_values = shot_from_grid_velocity(
                    grid_db_manager,
                    target_x,
                    target_y,
                    use_cw=use_cw,
                )
                if shot_values is not None:
                    (
                        translational_velocity,
                        shot_angle,
                        angular_velocity,
                        grid_x,
                        grid_y,
                        raw_vx,
                        raw_vy,
                        raw_w,
                    ) = shot_values
                    logger.info(
                        "DB lookup rotation=%s target=(%.3f, %.3f) -> grid=(%.1f, %.1f)",
                        rotation_label,
                        target_x,
                        target_y,
                        grid_x,
                        grid_y,
                    )
                else:
                    logger.warning(
                        "Grid velocity not found for rotation=%s target=(%.3f, %.3f). Use fallback shot.",
                        rotation_label,
                        target_x,
                        target_y,
                    )
                    translational_velocity = 2.39758149
                    shot_angle = 1.516130711
                    angular_velocity = -1.570796327 if use_cw else 1.570796327

                logger.info(
                    "Shot %d: target=(%.3f, %.3f), shot=(v=%.6f, angle=%.6f, omega=%.6f)",
                    next_shot_index,
                    target_x,
                    target_y,
                    translational_velocity,
                    shot_angle,
                    angular_velocity,
                )

                await client.send_shot_info(
                    translational_velocity=translational_velocity,
                    shot_angle=shot_angle,
                    angular_velocity=angular_velocity,
                )
                # なお、デジタルカーリング3で使用されていた、(vx, vy, rotation(cw または ccw))での送信も可能です。
                # vx = 0.0
                # vy = 2.33
                # rotation = "cw"
                # await client.send_shot_info_dc3(
                #     vx=vx,
                #     vy=vy,
                #     rotation=rotation,
                # )
    except Exception as e:
        client.logger.error(f"Unexpected error in main loop: {e}")
    finally:
        # 試合終了後、あるいはエラー時に溜まったログをファイルに書き出す
        # ファイル名（チーム名や時刻）の生成やディレクトリ作成はライブラリが自動で行います
        client.save_log_file()

if __name__ == "__main__":
    asyncio.run(main())
