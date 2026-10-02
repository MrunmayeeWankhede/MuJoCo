# Make the skeleton hit the scuba :D

A MuJoCo learning project that now includes a full-body anatomical MyoSim dance,
a simpler muscle skeleton, and PPO training for the simpler model.

![Anatomical scuba dance](results/myo_scuba/myo_scuba_preview.gif)

## Run the real skeleton on your Mac

Run these commands from the repository folder. For a fresh setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-myo.txt
```

Start the dance:

```bash
.venv/bin/mjpython myo_scuba.py --seconds 20
```

Add `--hide-muscles` to see just the bones. The right hand stays near the nose;
the left hand sweeps sideways with its palm aimed inward toward the chest.
On Linux/Windows, use `python myo_scuba.py --seconds 20` in the activated environment.

The official MyoSim model has 416 muscle actuators. Muscle excitation drives
physics; the pelvis is supported. This is an approximate dance with a muscle
controller, and can run slower than real time on some Macs. Exact nose pinching,
free balance, and a learned anatomical dance are still future work.

## Check, render, or train

```bash
# Physics without opening the viewer
.venv/bin/python myo_scuba.py --headless --seconds 12

# All tests (anatomical tests require requirements-myo.txt)
.venv/bin/python -m unittest discover -s tests -v

# Generate anatomical previews; requires a working graphics context
.venv/bin/python render_myo_scuba.py

# Watch the simpler muscle skeleton
.venv/bin/mjpython scuba.py --seconds 20

# Train PPO on the simpler skeleton, then watch it
.venv/bin/python train_scuba.py --steps 500000 --seed 7
.venv/bin/mjpython scuba.py --policy results/scuba/scuba_ppo.zip --seconds 20
```

For just the simpler skeleton, install `requirements.txt`. PPO checkpoints and
raw rollout arrays stay local; preview images and measured results are tracked.
The PPO policy applies to the simpler skeleton. The anatomical demo uses its
own muscle controller.

## Repository layout

```text
MuJoCo/
├── myo_scuba.py             # Anatomical dance controller and viewer
├── scuba.py                 # Simple dance controller and Gymnasium task
├── train_scuba.py           # PPO training for the simple skeleton
├── render_myo_scuba.py      # Anatomical previews and hand close-ups
├── render_scuba.py          # Simple skeleton previews
├── models/                 # Custom MuJoCo model XML
├── examples/
│   ├── basics/             # First box and pendulum experiments
│   ├── myosuite/           # Elbow, arm, hand, and muscle experiments
│   └── analysis/           # Joint inspection and muscle heatmaps
├── tests/                  # Physics, motion, and palm-orientation checks
├── results/
│   ├── myo_scuba/          # Anatomical previews, metrics, and local rollouts
│   ├── scuba/              # Simple previews, PPO metrics, and local checkpoints
│   └── analysis/           # Muscle-to-joint heatmaps
└── docs/                   # Detailed demo guides and learning notes
```

The main launch scripts stay at the top level for convenient Mac commands.
Original learning scripts are grouped under [examples](examples/README.md).
Generated outputs go into the corresponding `results/` folder regardless of
which directory you launch the main scripts from. The anatomical model assets
come from the installed MyoSim package.

## Guides and results

- [Anatomical model and controller](docs/anatomical-scuba.md)
- [Simplified model and PPO training](docs/simplified-scuba.md)
- [Learning progress and notes](docs/learning.md)
- [Latest anatomical playback metrics](results/myo_scuba/myo_scuba_metrics.json)
- [PPO development-run metrics](results/scuba/scuba_ppo.json)
