import asyncio
import json
import numpy as np
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FCV1_MAPPING_TABLE_PATH = PROJECT_ROOT / "external" / "FCV1_mapping_table"
if FCV1_MAPPING_TABLE_PATH.exists() and str(FCV1_MAPPING_TABLE_PATH) not in sys.path:
    sys.path.insert(0, str(FCV1_MAPPING_TABLE_PATH))

from load_secrets import username, password
from dc4client.dc_client import DCClient
from dc4client.send_data import TeamModel, MatchNameModel, PositionedStonesModel

formatter = logging.Formatter(
    "%(asctime)s, %(name)s : %(levelname)s - %(message)s"
)

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

    try:
        async for state_data in client.receive_state_data():
            # ゲーム終了の判定
            if (winner_team := client.get_winner_team()) is not None:
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
                #await asyncio.sleep(2)
                
                #translational_velocity = 2.3
                #angular_velocity = np.pi / 2
                #shot_angle = np.pi / 2

                # ボタン目掛けて投げる
                translational_velocity = 2.39758149
                angular_velocity = -1.570796327
                shot_angle = 1.516130711

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
