"""Supported, simplified muscle skeleton: coordinated scuba and PPO tracking task."""
from pathlib import Path
import argparse
import json
import time

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "scuba.xml"


def reference(t, frequency=0.5):
    """Scuba interpretation: nose-covering right hand, left sweep, alternating knees."""
    phase = 2 * np.pi * frequency * t
    s, c = np.sin(phase), np.cos(phase)
    # Hold the left arm forward with a fixed elbow; one shoulder axis sweeps
    # horizontally. Phase-shifted shoulder/elbow oscillations create a loop.
    return np.array([0.12*s, 0.08*c, -0.15,
                     -1.75, -0.07, 1.16, 1.99,
                     -np.pi/2, 0.12 + 0.45*s, 0.0, 0.25,
                     0.25*s, 0.35 + 0.25*s, -0.15*s,
                     -0.25*s, 0.35 - 0.25*s, 0.15*s])


class ScubaEnv(gym.Env):
    """Actions are direct muscle excitations in [0, 1], with activation dynamics."""
    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=8.0, frequency=0.5):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_seconds = episode_seconds
        self.frequency = frequency
        self.action_space = gym.spaces.Box(0, 1, (self.model.nu,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            (3*self.model.nq + self.model.na + 2,), np.float32)
        self.steps = 0

    def target(self):
        return reference(self.data.time, self.frequency)

    def _obs(self):
        p = 2*np.pi*self.frequency*self.data.time
        return np.concatenate([self.data.qpos, self.data.qvel,
            self.data.act, self.target(), [np.sin(p), np.cos(p)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = reference(0, self.frequency)
        if options and options.get("randomize"):
            self.data.qpos[:] += self.np_random.uniform(-0.03, 0.03, self.model.nq)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Expected one finite excitation per muscle")
        self.data.ctrl[:] = np.clip(action, 0, 1)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.steps += 1
        error = float(np.mean((self.data.qpos-self.target())**2))
        reward = float(np.exp(-8*error) - 0.01*np.mean(action**2))
        failed = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        return self._obs(), reward, failed, self.steps*self.dt >= self.episode_seconds, {"tracking_mse": error}

    def scripted_action(self):
        # Map desired joint torque to opposing muscles using current moment arms
        # and MuJoCo's current muscle force gains, including force-length effects.
        eps = 0.001
        velocity = (reference(self.data.time+eps, self.frequency)-self.target())/eps
        torque = 45*(self.target()-self.data.qpos) + 4*(velocity-self.data.qvel)
        torque += self.data.qfrc_bias
        action = np.full(self.model.nu, 0.02)
        for j in range(self.model.nv):
            # Two fixed tendon muscles per hinge; negative coefficient gives positive torque.
            a = 2*j + (0 if torque[j] >= 0 else 1)
            gain = mujoco.mju_muscleGain(self.data.actuator_length[a],
                self.data.actuator_velocity[a], self.model.actuator_lengthrange[a],
                self.model.actuator_acc0[a], self.model.actuator_gainprm[a, :9])
            capacity = abs(gain*0.04)
            action[a] = np.clip(abs(torque[j])/max(capacity, 1e-6)+0.02, 0, 1)
        return action.astype(np.float32)


def rollout(env, seconds, policy=None, viewer=None):
    env.reset()
    positions, targets, actions, errors = [], [], [], []
    for _ in range(int(np.ceil(seconds/env.dt))):
        start = time.monotonic()
        action = env.scripted_action() if policy is None else policy.predict(env._obs(), deterministic=True)[0]
        _, _, terminated, _, info = env.step(action)
        positions.append(env.data.qpos.copy())
        targets.append(env.target())
        actions.append(action)
        errors.append(info["tracking_mse"])
        if terminated:
            raise RuntimeError("Simulation became non-finite")
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0, env.dt-(time.monotonic()-start)))
    return np.asarray(positions), np.asarray(targets), np.asarray(actions), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"scuba"/"scuba_rollout.npz")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    env = ScubaEnv()
    policy = None
    if args.policy:
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy)
        if (policy.action_space.shape != env.action_space.shape or
                policy.observation_space.shape != env.observation_space.shape):
            parser.error("This checkpoint uses the old skeleton. Retrain with train_scuba.py.")
    if args.headless:
        q, target, action, errors = rollout(env, args.seconds, policy)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth = 20
            viewer.cam.elevation = -10
            viewer.cam.distance = 3.5
            viewer.cam.lookat[:] = [0, 0, 1.2]
            q, target, action, errors = rollout(env, args.seconds, policy, viewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, qpos=q, target=target, excitation=action, dt=env.dt)
    print(json.dumps({"steps": len(errors), "tracking_rmse_rad": float(np.sqrt(np.mean(errors))),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
