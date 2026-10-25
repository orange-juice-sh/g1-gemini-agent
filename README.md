# Controlling Unitree G1 with Natural-Language Instructions Using Gemini 4 Argon

English | [日本語](README-ja.md)

Final project for the Physical AI course (Applied 1).
A humanoid Unitree G1 with arms, simulated in MuJoCo, is driven only by Japanese instructions.
Gemini 4 Argon, the latest flagship reasoning model, acts as the "brain", and Unitree's official pretrained locomotion policy acts as the "cerebellum".
The AI plans and executes tasks that make use of a human-shaped body in a human workplace: greeting at the reception desk, giving directions after reading a sign, finding and carrying a lost item, and pressing an elevator button.

> Example: "There should be a red lost box in the meeting area. Find it, carry it to the reception desk, and put it on the counter."
> → The AI composes `walk_to(lounge)` → re-plans by side-stepping when the safety layer stops it → `find_object(red)` → `pick_up(red)` →
> `walk_to(reception)` → `put_down()` by itself.

## Architecture

```
 Japanese instruction (+ notes on what earlier instructions found out)
     │
 ┌───▼──────────────────────────────────────────────────┐
 │ High level: AI agent (Gemini 4 Argon, etc.)            │ observe → plan → act → check loop; re-plans from failure reasons
 │  Move  : walk_to / walk / turn / scan_surroundings     │ judges people, text and objects from the head-camera image
 │  Vision: look (image) / find_object / locate_point     │
 │  Arms  : gesture / press_button / pick_up / put_down   │
 │  Speech: say (voiced with VOICEVOX)                    │
 └───┬──────────────────────────────────────────────────┘
     │ skill calls (g1_agent/skills.py, 10 Hz)
 ┌───▼──────────────────────────────────────────────────┐
 │ Skill layer: path following, color detection + depth   │
 │   for 3D positions, arm IK (damped least squares)      │
 └───┬──────────────────────────────────────────────────┘
     │ velocity command (vx, vy, ωz) and arm joint targets
 ┌───▼──────────────────────────────────────────────────┐
 │ Safety layer: virtual LiDAR collision avoidance        │ obstacle within 0.7 m ahead → stop translation
 │   (independent of the AI), fall detection              │
 └───┬──────────────────────────────────────────────────┘
     │
 ┌───▼──────────────────────────────────────────────────┐
 │ Low level: pretrained locomotion policy (LSTM, 50 Hz,  │ motion.pt for G1 from unitree_rl_gym (re-implemented in NumPy)
 │   12 leg DoF) + squat policy (trained in Isaac Lab)    │
 │   + PD control of waist and arms, PD at 500 Hz         │
 └───┬──────────────────────────────────────────────────┘
     ▼
  Unitree G1 in MuJoCo (with arms, 29 DoF, locked waist) + a furnished office lobby
```

## Setup (Docker)

Assumes a machine with an NVIDIA GPU and the NVIDIA Container Toolkit (without a GPU, everything except rendering works with `--mock`).

```bash
git clone https://github.com/orange-juice-sh/g1-gemini-agent.git && cd g1-gemini-agent
cp .env.example .env                                  # write GEMINI_API_KEY=...
sed -i "s/^LOCAL_UID=.*/LOCAL_UID=$(id -u)/; s/^LOCAL_GID=.*/LOCAL_GID=$(id -g)/" .env
docker compose up -d --build
docker compose exec dev python scripts/fetch_assets.py   # download and convert CC0 furniture and textures (about 31 MB)

docker compose exec dev python run.py --mock "受付まで行ってください"   # check without an API key
```

The repository is mounted at `/workspace` in the container, so recordings and logs stay on the host.
On a machine without a display, use `--record` instead of `--viewer`.

## Live mode (used for the submitted video)

```bash
docker compose --profile tts up -d             # optional: voice for speech
docker compose exec dev python run.py --live --tts --scenario video
```

Forward the port with `ssh -L 8765:localhost:8765 <user>@<server>` and open http://localhost:8765.
The G1 moves in the browser in real time. Without `--scenario`, you can type instructions in the terminal.
The four instructions in the video (same as evaluation tasks D1–D4):

1. 休憩スペースにある赤色の四角い物体を受付に持って行ってください。 (Bring the red square object in the lounge to the reception desk.)
2. 休憩スペースにある青色の丸い物体を受付に持って行ってください。 (Bring the blue round object in the lounge to the reception desk. It is on the lower shelf, so the robot squats.)
3. 受付の人に手を振り、忘れ物を届けたことを伝えてください。 (Wave to the receptionist and say the lost items have been delivered.)
4. エレベーターのスイッチを押して、再度充電スペースに戻ってください。 (Press the elevator button and return to the charging area.)

## Web UI

```bash
docker compose exec dev python webui.py
```

Open http://localhost:8765 in the same way. Choose a brain (Argon / ER 2 / Flash / Qwen / rule-based), type an instruction, and watch the robot.
The UI shows the tool-call history, reports, and the API cost per instruction. Recordings are saved to `videos/webui-*.mp4`.

## Usage

Prefix each command with `docker compose exec dev`.

```bash
python run.py "受付に行って、誰かいるか確認してください"       # one instruction (default brain: Gemini 4 Argon)
python run.py                                                 # interactive mode
python run.py --scenario demo --tts --record videos/demo.mp4 --speed 2
python run.py --scenario demo --tts --replay logs/run-XXXX.json --record videos/demo.mp4 --speed 2   # re-render a log without the API
python run.py --task T20 --record videos/T20.mp4 --speed 2    # run an evaluation task

python run.py --model gemini:gemini-robotics-er-2-preview "会議室の前に行って"
python run.py --local "会議室の前に行って"                    # local Qwen3-VL
python run.py --mock "会議室の前に行って"                     # rule-based
```

Run logs (tool calls, API latency, tokens) are saved as JSON in `logs/`.

## Brains

| Option | Model | Role |
|---|---|---|
| `--model gemini` (default) | Gemini 4 Argon | Latest flagship reasoning model (for multi-step autonomous agents) |
| `--model gemini:gemini-robotics-er-2-preview` | Gemini Robotics-ER 2 | Model for robotics |
| `--model gemini:gemini-3.8-flash` | Gemini 3.8 Flash | Cheaper model of the same family |
| `--local` | Qwen3-VL-30B-A3B (FP8) | Open VLM on a local GPU (`docker compose --profile local up -d`) |
| `--mock` | Rule-based | Fixed procedures from keywords (evaluation baseline) |

All brains use exactly the same agent, tools, and skills (`g1_agent/gemini_llm.py` and `local_llm.py` convert the API formats).
The Claude API (`--model claude-...`) is also supported.

## Evaluation

```bash
python eval.py --models argon --trials 3 --workers 6
python eval.py --models er2 flash --trials 2 --workers 6
python eval.py --models qwen --trials 3 --workers 8
python eval.py --models rule
python plot_results.py $(cat results/final/SOURCES.txt) --out results/final
```

Tasks in `g1_agent/tasks.py` run in parallel. The success rate (with a Wilson 95% confidence interval), API latency, tool calls, and cost are saved to `results/eval-*/`.
Success is judged automatically from the final simulation state and the run log.

| ID | Type | Instruction (summary) | What it tests |
|---|---|---|---|
| T1–T2 | Move | Go to reception / meeting room | Re-planning after a safety stop |
| T3–T4 | Search | Find a red / blue box | Looking around |
| T5–T6 | Talk | Greet the receptionist / lead to the meeting room | Speaking at a place |
| T7–T8 | Vague | "I'm tired" / "Battery is low" | Reading intent |
| T9 | Vision | Speak to the guest in orange | Telling people apart from the image |
| T10 | Text | Read the sign and go to the meeting place | Reading Japanese text |
| T11 | Vision | Count the small boxes in the lounge | Counting |
| T12 | Change | A 4.4 m partition appears | Persistent re-planning |
| T13 | Change | The lost item is not where it was said to be | Widening the search |
| T14 | Long | Greet → report → return to charge | Long procedures |
| T15 | Search | Find a green ball (confusable with a plant) | Discrimination |
| T16–T19 | Arms | Point the way, wave, call the elevator, point after moving | Using the humanoid body |
| T20 | Arms | Carry a lost item to the counter | Lift, carry, place |
| D1–D4 | Demo | The four instructions of the video | Choosing by color and shape, squatting, two-step instructions |

Main results (T1–T20, `results/analysis.md`, figures in `results/final/`):

| Model | Success [95% CI] | Avg. latency | Cost per success |
|---|---|---|---|
| Gemini Robotics-ER 2 | 100% (40/40) [91–100%] | 4.8 s | $0.035 |
| Gemini 4 Argon | 95% (57/60) [86–98%] | 17.6 s | $0.058 |
| Gemini 3.8 Flash | 90% (36/40) [77–96%] | 5.7 s | $0.043 |
| Qwen3-VL (local) | 70% (42/60) [57–80%] | 0.7 s | — |
| Rule-based | 30% (6/20) [15–52%] | — | — |

## Squatting to pick items from a low shelf (RL in Isaac Lab)

The official locomotion policy cannot lower the hips, so we trained a separate policy that stops, squats to a commanded height, and stands up.
The robot switches to it only when picking items from the lower shelf (about 0.5 m high).

- Training: PPO (rsl_rl) in Isaac Lab 2.3.1, 4096 environments, with the same body model as MuJoCo (g1_29dof_lock_waist)
- Network: MLP (74 observations → 256-128-128 → 13 actions, ELU), run with NumPy in MuJoCo (`g1_agent/squat_policy.py`)
- The 13 design iterations (v1–v13) are recorded in `results/squat/README.md`. The keys were a strong penalty on lifting the feet (v10)
  and starting 70% of the episodes from states collected in MuJoCo at the moment of switching from walking (v13, `scripts/collect_handoff_states.py`)
- The trained policy is `lab/exported/squat_policy.npz`. In the same flow as the video (`scripts/test_pick_demo_flow.py`), it picked the blue ball 12/12 times

```bash
mkdir -p lab/assets
docker compose exec dev cp -r /opt/unitree_rl_gym/resources/robots/g1_description lab/assets/   # G1 model (URDF) for Isaac Lab
docker compose exec dev python scripts/collect_handoff_states.py
docker compose --profile lab up -d lab
docker compose exec lab bash /project/lab/run.sh train --num_envs 4096 --max_iterations 2000
docker compose exec -w /project/lab lab /workspace/isaaclab/isaaclab.sh -p export.py logs/rsl_rl/<experiment>/<date>/model_XXXX.pt
```

`lab/g1_squat/env_cfg.py` matches the final configuration. The final policy was trained by resuming across versions (v10 variant B → v11 → v12 → v13, about 2,000 updates in total), so training from scratch may give slightly different results.

## Files

| File | Contents |
|---|---|
| `run.py` | CLI (single, interactive, scenarios, tasks, recording, live mode) |
| `eval.py` / `plot_results.py` | Evaluation / figures |
| `webui.py` | Web UI |
| `g1_agent/agent.py` | Tool-use loop, tool definitions, system prompt |
| `g1_agent/skills.py` | Skills (moving, vision, gestures, buttons, carrying, speech) |
| `g1_agent/sim.py` | MuJoCo scene, locomotion and squat policies, arm PD and IK, safety layer |
| `g1_agent/tasks.py` / `world.py` | Evaluation tasks / lobby environment |
| `g1_agent/recorder.py` | Recording (camera switching, captions, minimap, cards, audio) |
| `g1_agent/policy.py` / `squat_policy.py` | NumPy implementations of the locomotion and squat policies |
| `g1_agent/gemini_llm.py` / `local_llm.py` / `mock_llm.py` | Adapters for Gemini, the local VLM, and the rule-based baseline |
| `lab/` | Squat training (Isaac Lab task `G1-Squat-v0`) and the trained policy |
| `scripts/` | Asset download, state collection for training, pick test |

## Highlights

1. **Hierarchy of AI and RL**: the AI decides the steps; bipedal balance is left to the pretrained RL policy.
2. **Closed-loop agent**: tool results include success or failure and the reason, so the AI re-plans. Findings carry over to the next instruction.
3. **Safety layer independent of the AI**: the virtual LiDAR stops the robot even if the AI makes a mistake or is slow.
4. **Using the humanoid body**: waving, pointing (IK toward a location), pressing buttons, and carrying with both hands. IK is re-solved every 0.1 s because the robot keeps stepping in place.
5. **AI vision + classical image processing**: the AI decides what is where in the image; 3D positions come from the depth image.
6. **Model comparison**: five brains on the same tools and tasks, with confidence intervals and cost.
7. **Own RL for squatting**: trained in Isaac Lab and switched with walking in MuJoCo. The sim-to-sim gap was closed by matching the state distribution at the moment of switching.
8. **Reproducibility**: Docker, asset scripts, and a rule-based brain that needs no API key.

## Simplifications and limitations

- Simulation only; not run on the real robot.
- Grasping is simplified: when both fingertips reach the sides of an item, it is attached between the hands.
- Low items are picked by squatting (down to a hip height of 0.50 m). Items directly on the floor cannot be reached. The robot cannot squat while walking.
- Object detection uses red, blue, green, and yellow color thresholds; other objects are handled by the AI pointing to image coordinates (`locate_point`).
- People do not move.
- The submitted video was recorded in live mode with Gemini 4 Argon. Some parts are sped up in editing (stated in the video and its description).

## Credits

- G1 model and pretrained locomotion policy: [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (BSD-3-Clause), fetched when building the Docker image
- Simulator: [MuJoCo](https://mujoco.org/); RL: [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause), [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause)
- References: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347; Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning", CoRL 2021, arXiv:2109.11978
- AI: [Gemini API](https://ai.google.dev/) (Gemini 4 Argon, Gemini Robotics-ER 2, Gemini 3.8 Flash), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) (Apache-2.0)
- Furniture: [Poly Haven](https://polyhaven.com/) (CC0): sofa_02, modern_arm_chair_01, modern_coffee_table_01, side_table_01, potted_plant_02, modern_wooden_cabinet, dining_chair_02
- Textures: [ambientCG](https://ambientcg.com/) (CC0): Tiles078, PaintedPlaster017, Wood092, WoodFloor064, Carpet012 (assets are not included; get them with `scripts/fetch_assets.py`)
- Speech synthesis: [VOICEVOX](https://voicevox.hiroshiba.jp/) (VOICEVOX:No.7, "announce" style; non-commercial use, credited in the video)

## License

MIT (see `LICENSE`)
