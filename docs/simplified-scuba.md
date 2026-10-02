# Simplified scuba and PPO training

Run the commands below from the repository root.


![Supported muscle skeleton doing a scuba-inspired motion](../results/scuba/scuba_preview.gif)

This adds a **simplified full-body MuJoCo skeleton**, with 17 hinge joints and
34 opposing muscle actuators connected through fixed tendons. MuJoCo simulates
muscle activation dynamics and force-length/velocity effects; the controller
commands muscle excitation, not joint positions. Both shoulders include a twist joint so a bent arm can reach around the chest.
Arm–torso collisions are enabled, including explicit upper-arm/spine and rib
contacts. The scripted motion keeps both arms clear of the torso.
The pelvis is fixed, the feet
have no contact forces, and the bones are primitive shapes. This is a learning
model, **not an anatomically validated MyoSuite full-body model**.

The hand-authored interpretation combines a right hand raised toward the face,
a left arm sweeping side to side at nearly constant height, torso sway, head nod,
and alternating bent knees. The left elbow and forward shoulder angle stay
fixed while one shoulder axis oscillates, avoiding the original circular hand path. It is an
approximation of the scuba, without a motion-capture reference. Balance, realistic
muscle paths, fingers pinching the nose, and a validated learned dance remain
future work. The earlier MyoSuite experiments are in `examples/myosuite/`.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

These requirements run the new scuba workflow. For the older experiments in `examples/myosuite/`, install `requirements-myo.txt` as well.

### Watch or check the scripted dance

On macOS use MuJoCo's Python launcher for the interactive viewer:

```bash
.venv/bin/mjpython scuba.py --seconds 20
```

On Linux/Windows, use `python scuba.py --seconds 20` from the activated environment.
Close the window to stop early. For physics without a graphics window:

```bash
python scuba.py --headless --seconds 12
python -m unittest discover -s tests -v
python render_scuba.py
```

`render_scuba.py` saves a PNG and a four-second looping GIF and requires a working
OpenGL context. Headless physics and training do not require graphics.
Rollouts save joint positions, target angles, muscle excitations, and timestep
to `results/scuba/scuba_rollout.npz`, and print tracking RMSE in radians.

### Train muscle coordination

```bash
python train_scuba.py --steps 500000 --seed 7
.venv/bin/mjpython scuba.py --policy results/scuba/scuba_ppo.zip --seconds 20
```

The Gymnasium task observes joint position/velocity, muscle activation, target
angles, and motion phase. PPO learns 34 muscle excitations using a tracking reward
with an excitation penalty. Episodes last eight seconds; the scripted PD
controller supplies a comparison, not policy actions during training. Each run
saves a checkpoint and JSON metrics comparing untrained, trained, and scripted
tracking. Increase `--steps` to experiment; training does not guarantee a good
dance. A 100k-step development run may be available locally in `results/scuba/scuba_ppo.zip`;
the checkpoint is ignored by Git. Measured results are in
[results/scuba/scuba_ppo.json](../results/scuba/scuba_ppo.json). The final RL milestone stays
unchecked because this supported-model baseline does not establish a realistic,
freely balancing scuba.

### New files

- `models/scuba.xml`: supported skeleton, opposing tendon muscles, floor and light.
- `scuba.py`: motion target, muscle controller, Gymnasium task, viewer and rollout CLI.
- `train_scuba.py`: reproducible PPO training and before/after evaluation.
- `render_scuba.py`: still and animated preview generator.
- `tests/test_scuba.py`: environment contract, target limits, multi-cycle physics, arm clearance, and collision checks.


Shoulder correction: checkpoints trained with the original 15-joint skeleton
are incompatible with the corrected 17-joint model. Retrain using the command
above; the local `results/scuba/scuba_ppo.zip` has been regenerated for this model.
