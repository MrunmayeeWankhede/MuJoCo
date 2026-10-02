"""Learn muscle excitation with PPO against the hand-authored scuba target."""
import argparse
from pathlib import Path
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scuba import ScubaEnv, ROOT, rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=500_000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', type=Path, default=ROOT/'results'/'scuba_ppo')
    args = p.parse_args()
    if args.steps <= 0:
        p.error('--steps must be positive')
    env = ScubaEnv()
    check_env(env)
    model = PPO('MlpPolicy', env, seed=args.seed, n_steps=1024,
                batch_size=64, learning_rate=3e-4, verbose=1, device='cpu')
    _, _, _, before = rollout(env, 8, model)
    model.learn(total_timesteps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    _, _, _, after = rollout(env, 8, model)
    _, _, _, scripted = rollout(env, 8)
    metrics = {'seed': args.seed, 'training_steps': model.num_timesteps,
               'untrained_rmse_rad': float(np.sqrt(np.mean(before))),
               'trained_rmse_rad': float(np.sqrt(np.mean(after))),
               'scripted_rmse_rad': float(np.sqrt(np.mean(scripted)))}
    args.output.with_suffix('.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == '__main__':
    main()
