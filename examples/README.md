# Learning experiments

These are the original exploratory scripts, grouped by topic. Run commands from
the repository root. Some files contain several experiments in sequence; read
their notes before launching them.

| Folder | Scripts | Topic |
| --- | --- | --- |
| `basics/` | `arm.py`, `pendulum_rhythm.py` | First MuJoCo models and sine-wave control |
| `myosuite/` | `myo_model.py`, `myo_arm.py`, `myo_hand.py` | Muscle-driven elbow, arm, and hand models |
| `myosuite/` | `hand_gesture.py`, `co_contraction.py`, `agonist_antagonist.py` | Coordinated activation and opposing muscles |
| `analysis/` | `arm_muscle_map.py`, `muscle_map_directional.py` | Muscle-to-joint movement experiments |
| `analysis/` | `joint_names.py`, `list_envs.py` | Inspect anatomical joints and available environments |
| `analysis/` | `visualize.py`, `arm_visualize.py` | Save muscle-to-joint heatmaps |

Install `requirements.txt` for the basic MuJoCo examples, `requirements-myo.txt`
for MyoSuite examples, or `requirements-analysis.txt` for the heatmap scripts:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
```

Example commands on macOS:

```bash
.venv/bin/mjpython examples/basics/pendulum_rhythm.py
.venv/bin/python examples/analysis/list_envs.py
.venv/bin/python examples/analysis/joint_names.py
.venv/bin/python examples/analysis/arm_visualize.py
```

The heatmaps are saved under `results/analysis/`. The current full-body dance is
launched with `myo_scuba.py`; see the [main README](../README.md).
