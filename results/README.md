# Simulation outputs

| Folder | Contents | Generator |
| --- | --- | --- |
| `myo_scuba/` | Anatomical PNG/GIF previews and hand close-ups | `render_myo_scuba.py` |
| `myo_scuba/` | Playback metrics and raw rollout arrays | `myo_scuba.py` |
| `scuba/` | Simple skeleton PNG/GIF previews | `render_scuba.py` |
| `scuba/` | Rollout arrays | `scuba.py` |
| `scuba/` | PPO checkpoint and evaluation metrics | `train_scuba.py` |
| `analysis/` | Muscle-to-joint heatmaps | `examples/analysis/visualize.py`, `arm_visualize.py` |

Commands are documented in the [main README](../README.md). PNG/GIF previews,
JSON metrics, and recorded dependency snapshots (`*_environment.txt`) are
tracked. Raw `.npz` rollouts, `.zip` checkpoints, and monitor CSVs are ignored.
Dependency snapshots record the environment used for past development runs;
use the root requirements files when installing.

The anatomical metrics describe the most recent completed or interrupted
playback, so the duration can differ from the animated preview. Playback,
rendering, and training overwrite their corresponding default output files.
