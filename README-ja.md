# Gemini 4 Argonを用いた自然言語指示によるUnitree G1の制御

[English](README.md) | 日本語

Physical AI 応用編1 最終課題。
最新のフラッグシップ推論モデル Gemini 4 Argon を「頭脳」に、Unitree 公式の学習済み歩行ポリシーを「小脳」にして、
MuJoCo 上の腕付きヒューマノイド G1 を日本語の指示だけで動かします。
受付での挨拶、案内板を読んでの道案内、忘れ物を探して運ぶ、エレベーターのボタンを押す、といった
「人の働く場所で、人の体の形を生かす」仕事を、AI が自分で手順を組み立てて実行します。

> 例：「打ち合わせの場所に赤い忘れ物の箱があるはずです。探して受付まで運び、受付カウンターの上に置いてください。」
> → AI が `walk_to(休憩スペース)` → 安全停止したら横に避けて再計画 → `find_object(red)` → `pick_up(red)` →
> `walk_to(受付)` → `put_down()` を自分で組み立てて実行します。

## 構成

```
 日本語の指示（＋前の指示で分かったことの記録）
     │
 ┌───▼──────────────────────────────────────────────────┐
 │ 高レベル：AI のエージェント（Gemini 4 Argon など）       │ 観測→計画→実行→確認 のループ。失敗理由を受け取って再計画
 │  移動: walk_to / walk / turn / scan_surroundings        │ 頭部カメラの画像を見て判断（人・文字・物）
 │  視覚: look(画像) / find_object / locate_point(画像の点) │
 │  腕  : gesture / press_button / pick_up / put_down      │
 │  発話: say（VOICEVOX で音声に）                          │
 └───┬──────────────────────────────────────────────────┘
     │ スキル呼び出し（g1_agent/skills.py、10 Hz）
 ┌───▼──────────────────────────────────────────────────┐
 │ スキル層：経路追従、色検出＋深度での3次元位置推定、       │
 │           腕の逆運動学（減衰付き最小二乗）、体の揺れへの追従│
 └───┬──────────────────────────────────────────────────┘
     │ 速度指令 (vx, vy, ωz)・腕の関節の目標角
 ┌───▼──────────────────────────────────────────────────┐
 │ 安全層：仮想 LiDAR による衝突回避（AI から独立）、転倒検知 │ 進行方向 0.7m 以内に障害物 → 並進停止
 └───┬──────────────────────────────────────────────────┘
     │
 ┌───▼──────────────────────────────────────────────────┐
 │ 低レベル：学習済み歩行ポリシー（LSTM、50 Hz、脚12自由度） │ unitree_rl_gym の G1 用 motion.pt（NumPy で再実装）
 │          + 腰・腕の PD 制御（15自由度）＋ PD（500 Hz）    │
 └───┬──────────────────────────────────────────────────┘
     ▼
  MuJoCo 上の Unitree G1（腕付き、29自由度・腰固定モデル）＋ 家具のあるオフィスのロビー
```

## セットアップ（Docker）

NVIDIA の GPU と NVIDIA Container Toolkit が使えるサーバーを想定しています（GPU がなくても `--mock` と描画以外は動きます）。

```bash
git clone https://github.com/orange-juice-sh/g1-gemini-agent.git && cd g1-gemini-agent
cp .env.example .env                                  # GEMINI_API_KEY=... を書き込む
sed -i "s/^LOCAL_UID=.*/LOCAL_UID=$(id -u)/; s/^LOCAL_GID=.*/LOCAL_GID=$(id -g)/" .env  # 自分のユーザーIDを設定
docker compose up -d --build                          # イメージを作ってコンテナを起動
docker compose exec dev python scripts/fetch_assets.py   # 家具・テクスチャ（CC0）を取得して変換（約31MB）

docker compose exec dev python run.py --mock "受付まで行ってください"   # API キーなしで動作確認
```

リポジトリはコンテナの `/workspace` にマウントされるので、録画やログはホスト側にそのまま残ります。
画面のないサーバーでは `--viewer` は使わず、`--record` で動画にして確認します。

## ブラウザから指示を出して動かす（Web UI）

```bash
docker compose up -d dev                       # ポート 8765 をサーバー自身（127.0.0.1）にだけ公開
docker compose --profile tts up -d             # （任意）発話を音声で聞く場合
docker compose exec dev python webui.py        # 起動（Ctrl+C で終了）
```

手元の PC で `ssh -L 8765:localhost:8765 <ユーザー名>@<サーバー>` を実行し、ブラウザで http://localhost:8765 を開きます。
日本語で指示を入力すると、選んだ頭脳（Argon / ER 2 / Flash / Qwen / ルールベース）が考えて、G1 が動く様子が実時間で映ります。
ツール呼び出しの履歴、報告、指示ごとの API 費用（計算値）が表示され、「録画」を押している間の映像は `videos/webui-*.mp4` に保存されます。

## ターミナルで指示を出しながらブラウザで見る（ライブモード。提出動画の撮影に使用）

```bash
docker compose up -d dev
docker compose --profile tts up -d             # （任意）発話を音声にする場合
docker compose exec dev python run.py --live --tts --scenario video   # 提出動画の4つの指示を順に実行
```

Web UI と同じく `ssh -L 8765:localhost:8765 <ユーザー名>@<サーバー>` でポートを転送し、http://localhost:8765 を開きます。
G1 の動きがブラウザに実時間で映ります。`--scenario` を付けなければ、ターミナルから自由に指示を入力できます。
提出動画の4つの指示は次のとおりです（評価タスク D1〜D4 と同じ）。

1. 休憩スペースにある赤色の四角い物体を受付に持って行ってください。
2. 休憩スペースにある青色の丸い物体を受付に持って行ってください。（棚の下の段にあり、しゃがんで取る）
3. 受付の人に手を振り、忘れ物を届けたことを伝えてください。
4. エレベーターのスイッチを押して、再度充電スペースに戻ってください。

## 使い方

以下のコマンドは、先頭に `docker compose exec dev` を付けて実行します。

```bash
# 1つの指示を実行（既定の頭脳は Gemini 4 Argon）
python run.py "受付に行って、誰かいるか確認してください"

# 提出動画のデモ（5場面）を、字幕・音声付きで録画（2倍速。発話の場面は等倍）
docker compose --profile tts up -d                     # 音声合成（VOICEVOX ENGINE）を起動
python run.py --scenario demo --tts --record videos/demo.mp4 --speed 2 --eval results/eval-...

# 提出動画の4つの指示を順に実行し、ブラウザに映す（ライブモード）
python run.py --live --tts --scenario video

# 実行ログを再生して録画し直す（API を使わない。判断は元の実行と同じで、映像の演出だけを作り直せる）
python run.py --scenario demo --tts --replay logs/run-XXXX.json --record videos/demo.mp4 --speed 2

# 評価タスクのシーンで1つ実行・録画（例: T20 忘れ物を運ぶ）
python run.py --task T20 --record videos/T20.mp4 --speed 2

# 頭脳を変える
python run.py --model gemini:gemini-robotics-er-2-preview "会議室の前に行って"   # Gemini Robotics-ER 2
python run.py --local "会議室の前に行って"                                      # ローカルの Qwen3-VL
python run.py --mock "会議室の前に行って"                                       # ルールベース
```

実行ログ（ツール呼び出し、API の遅延、トークン数）は `logs/` に JSON で保存されます。

## 頭脳にするモデル

| 指定 | モデル | 位置づけ |
|---|---|---|
| `--model gemini`（既定） | Gemini 4 Argon | 最新のフラッグシップ推論モデル（多段階の自律エージェント向け） |
| `--model gemini:gemini-robotics-er-2-preview` | Gemini Robotics-ER 2 | ロボット向けのモデル（Gemini 3.5 Flash ベース） |
| `--model gemini:gemini-3.8-flash` | Gemini 3.8 Flash | 同じ系列の安いモデル |
| `--local` | Qwen3-VL-30B-A3B（FP8） | GPU 上で動くオープンな VLM（`docker compose --profile local up -d`） |
| `--mock` | ルールベース | キーワードから決まった手順を組み立てる（評価のベースライン） |

どのモデルでも、エージェント・ツール・スキルはまったく同じものを使います
（`g1_agent/gemini_llm.py` と `local_llm.py` が API の形式を変換）。Claude API（`--model claude-...`）にも切り替えられます。

## 定量評価

```bash
python eval.py --models argon --trials 3 --workers 6
python eval.py --models er2 flash --trials 2 --workers 6
python eval.py --models qwen --trials 3 --workers 8
python eval.py --models rule
python plot_results.py $(cat results/final/SOURCES.txt) --out results/final
```

タスク（`g1_agent/tasks.py`）を並列に実行し、成功率（Wilson の 95% 信頼区間つき）・API 遅延・
ツール呼び出し回数・費用（キャッシュの割引込み）・成功1回あたりの費用を `results/eval-*/` に保存します。
成功かどうかは、シミュレーションの最終状態と実行ログから機械的に判定します。

| ID | 種類 | 指示（要約） | 試すこと |
|---|---|---|---|
| T1〜T2 | 移動 | 受付・会議室へ行く | 安全停止からの再計画 |
| T3〜T4 | 探索 | 赤い箱・青い箱を探して前まで行く | 見回して探す |
| T5〜T6 | 対話 | 受付の人に挨拶・会議室へ先導して伝える | 場所での発話 |
| T7〜T8 | 曖昧 | 「疲れた」「バッテリーが少ない」 | 意図の読み取り |
| T9 | 視覚 | オレンジの服のお客さんに声をかける | 色の道具で探せない人を画像で見分け、locate_point で近づく |
| T10 | 文字 | 案内板を読んで打ち合わせの場所へ行く | 日本語の文字の読み取り |
| T11 | 視覚 | 休憩スペースの小さな箱を数える | 見回して数える |
| T12 | 変化 | 移動中に幅 4.4m の仕切りが現れる | 粘り強い再計画 |
| T13 | 変化 | 忘れ物が聞いていた場所にない | 探す範囲を広げる |
| T14 | 長手順 | 挨拶 → 会議室で報告 → 充電に戻る | 長い手順 |
| T15 | 探索 | 緑のボールを探す（緑の観葉植物と紛らわしい） | 見分け |
| T16〜T19 | 腕 | 指差しで道案内・手を振って挨拶・エレベーターを呼ぶ・移動後に指差す | 人型の体を使う |
| T20 | 腕 | 忘れ物を受付まで運んでカウンターに置く | 持ち上げ・運搬・設置 |
| D1〜D4 | デモ | 提出動画と同じ4つの指示（撮影用のロビー scene_shelf） | 色と形の言葉での物の選択、しゃがんで取る、2段階の指示 |

本番評価（T1〜T20）の結果（集計と失敗の分析は `results/analysis.md`、グラフは `results/final/`）:

| モデル | 成功率 [95%CI] | 平均応答 | 成功1回あたり |
|---|---|---|---|
| Gemini Robotics-ER 2 | 100%（40/40）[91–100%] | 4.8 秒 | $0.035 |
| Gemini 4 Argon | 95%（57/60）[86–98%] | 17.6 秒 | $0.058 |
| Gemini 3.8 Flash | 90%（36/40）[77–96%] | 5.7 秒 | $0.043 |
| Qwen3-VL（ローカル） | 70%（42/60）[57–80%] | 0.7 秒 | — |
| ルールベース | 30%（6/20）[15–52%] | — | — |

## しゃがんで棚の下の段の物を取る（Isaac Lab での強化学習）

公式の歩行ポリシーは腰を落とせないため、「止まって指定の高さまでしゃがみ、立ち上がる」ポリシーだけを自分で学習し、
棚の下の段（高さ約 0.5m）の物を取るときだけ歩行ポリシーから切り替えます。

- 学習：Isaac Lab 2.3.1 の PPO（rsl_rl）、4096 環境。体のモデルは MuJoCo と同じ g1_29dof_lock_waist（関節の armature などもそろえる）
- ネットワーク：MLP（観測 74 → 256-128-128 → 行動 13、ELU）。MuJoCo 側では NumPy で推論（`g1_agent/squat_policy.py`）
- 13 回の設計の見直し（v1〜v13）の経過は `results/squat/README.md`。決め手は、足を浮かせることへの強い罰（v10）と、
  試行の 7 割を MuJoCo で集めた「歩行から切り替える瞬間の状態」から始めること（v13、`scripts/collect_handoff_states.py`）
- 学習済みのポリシーは `lab/exported/squat_policy.npz`（v13 の 1800 回目）。本番と同じ流れの試験（`scripts/test_pick_demo_flow.py`）で、
  下の段の青いボールを 12/12 回取れた

```bash
mkdir -p lab/assets
docker compose exec dev cp -r /opt/unitree_rl_gym/resources/robots/g1_description lab/assets/   # Isaac Lab 用の G1 のモデル（URDF）
docker compose exec dev python scripts/collect_handoff_states.py              # 歩行から切り替える瞬間の状態を MuJoCo で集める（lab/handoff_states.npz）
docker compose --profile lab up -d lab
docker compose exec lab bash /project/lab/run.sh train --num_envs 4096 --max_iterations 2000   # 学習（lab/logs に保存）
docker compose exec -w /project/lab lab /workspace/isaaclab/isaaclab.sh -p export.py logs/rsl_rl/<実験名>/<日時>/model_XXXX.pt  # NumPy 用に書き出す
```

`lab/g1_squat/env_cfg.py` の設定は最終版と同じです。最終版は、設計を変えながら前の版の続きから学習を重ねたもの（v10 案B → v11 → v12 → v13、合計約 2,000 回の更新）なので、一から学習した場合は結果が少し変わることがあります。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `run.py` | 実行用 CLI（単発・対話・デモ・タスク・録画・音声） |
| `eval.py` / `plot_results.py` | 定量評価 / グラフ |
| `webui.py` | Web UI |
| `g1_agent/agent.py` | tool use のループ、ツール定義、システムプロンプト、作業の記録 |
| `g1_agent/skills.py` | スキル層（移動、視覚、腕のジェスチャー・ボタン・持ち運び、発話） |
| `g1_agent/sim.py` | MuJoCo のシーン構築、歩行ポリシー、腕の PD と IK、安全層、転倒検知 |
| `g1_agent/tasks.py` | 評価タスク（指示・シーン・成功条件） |
| `g1_agent/world.py` | ロビー環境（場所・家具・探索対象） |
| `g1_agent/recorder.py` | 録画（カメラの自動切り替え、字幕、頭部カメラ、ミニマップ、章立て・構成図・まとめのカード、音声の合成） |
| `g1_agent/policy.py` | 学習済み LSTM 歩行ポリシーを PyTorch なしで実行する NumPy 実装 |
| `g1_agent/squat_policy.py` | 自分で学習したしゃがむポリシー（MLP）の NumPy 実装 |
| `lab/` | しゃがむ動作の学習（Isaac Lab のタスク `G1-Squat-v0`、書き出し、学習済みのポリシー） |
| `g1_agent/gemini_llm.py` / `local_llm.py` / `mock_llm.py` | Gemini（Interactions API）・ローカル VLM・ルールベースの変換層 |
| `g1_agent/tts.py` | VOICEVOX での音声合成 |
| `scripts/fetch_assets.py` | 家具・テクスチャ（CC0）の取得と glTF → OBJ の変換 |

## 工夫した点

1. **AI と強化学習の階層化**：AI は手順を決め、二足歩行のバランスは学習済みの RL ポリシーに任せる。
2. **閉ループのエージェントと作業の記録**：ツールの結果を「成功・失敗とその理由」付きで返して再計画させる。前の指示で分かったこと（案内板に書いてあった場所など）を次の指示に引き継ぐ。
3. **AI から独立した安全層**：AI の判断ミスや遅延があっても、仮想 LiDAR が障害物を見つけたら止まる。
4. **人型の体を生かす動作**：腕付きの G1 で、手を振る・指差す（IK で場所の方向へ腕を向ける）・ボタンを押す・両手で物を運ぶ。歩行ポリシーが止まっていても足踏みで揺れるため、0.1 秒ごとに IK を解き直して手先を追従させた。
5. **AI の視覚と従来型の画像処理の組み合わせ**：「何がどこに写っているか」は AI が画像を見て判断し（locate_point で画像の点を指す）、3次元の位置は深度画像から計算する。
6. **モデルの比較**：最新の推論モデル・ロボット向けモデル・安いモデル・ローカル VLM・ルールベースを、同じツールと同じタスクで、信頼区間と費用つきで比べた。
7. **再現しやすさ**：Docker、素材の取得スクリプト、API キーなしで動くルールベース。
8. **しゃがむ動作を自分で強化学習**：公式の歩行ポリシーではできない「しゃがんで棚の下の物を取る」を Isaac Lab で学習し、MuJoCo に移して歩行と切り替えて使う。シミュレータ間の差（sim2sim）を、切り替える瞬間の状態の分布をそろえることで埋めた。
9. **実機への移行を意識**：歩行ポリシーは Unitree が実機用に配布しているもの。スキル層の指令は実機 SDK の指令に置き換えられる設計（今回は安全面から実機では実行していない）。

## 簡略化と制約

- **シミュレーションのみ**で、実機では動かしていない。
- **物をつかむ動作は簡略化**している：両手の指先が物の側面に届いたら、物を両手の間に固定して一緒に動かす（指の力や摩擦は計算していない）。
- 歩行ポリシーは腰を曲げられないため、低い物はしゃがむポリシーに切り替えて取る（腰の高さ 0.50m まで。床に直接置いた物には届かない）。
- しゃがむポリシーは止まっているときだけ使う（歩きながらはしゃがめない）。
- 物体検出は赤・青・緑・黄の色しきい値。色以外の物は、AI が画像の中の点を指す（locate_point）ことで扱う。
- 人は動かない（マネキン風の見た目）。
- 提出動画は、ライブモード（`run.py --live`）で Gemini 4 Argon に4つの指示を出して撮影した。移動の場面などは編集で倍速にしている（動画内・概要欄に明記）。

## クレジット

- G1 のモデルと学習済み歩行ポリシー：[unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)（BSD-3-Clause）。リポジトリには含めず、Docker イメージの作成時に取得します。
- シミュレータ：[MuJoCo](https://mujoco.org/)、強化学習：[Isaac Lab](https://github.com/isaac-sim/IsaacLab)（BSD-3-Clause）・[rsl_rl](https://github.com/leggedrobotics/rsl_rl)（BSD-3-Clause）
- 参考文献：Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347 / Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning", CoRL 2021, arXiv:2109.11978
- AI：[Gemini API（Gemini 4 Argon・Gemini Robotics-ER 2・Gemini 3.8 Flash）](https://ai.google.dev/)、[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)（Apache-2.0）
- 家具の 3D モデル：[Poly Haven](https://polyhaven.com/)（CC0）の sofa_02、modern_arm_chair_01、modern_coffee_table_01、side_table_01、potted_plant_02、modern_wooden_cabinet、dining_chair_02
- 床・壁・木目・カーペットのテクスチャ：[ambientCG](https://ambientcg.com/)（CC0）の Tiles078、PaintedPlaster017、Wood092、WoodFloor064、Carpet012
- 素材はリポジトリに含めず、`scripts/fetch_assets.py` で取得・変換します（素材がなくても単純な形の見た目で動きます）
- 音声合成：[VOICEVOX](https://voicevox.hiroshiba.jp/)（音声: VOICEVOX:No.7、「アナウンス」スタイル。No.7 の規約により非商用で利用し、動画内にクレジットを表示）

## ライセンス

MIT（`LICENSE` を参照）
