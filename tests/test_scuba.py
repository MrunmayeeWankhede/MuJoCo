import unittest
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env
from scuba import ScubaEnv, reference, rollout


class ScubaTests(unittest.TestCase):
    def test_environment_contract(self):
        env = ScubaEnv(episode_seconds=0.04)
        check_env(env, warn=False)
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.model.nu, 34)
        self.assertEqual(env.model.nq, 17)
        self.assertTrue(np.all(env.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE))
        self.assertTrue(np.all(env.model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON))
        _, _, terminated, truncated, _ = env.step(np.zeros(34))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(env.step(np.zeros(34))[3])
        env.reset()
        self.assertEqual(env.data.time, 0)
        with self.assertRaises(ValueError):
            env.step(np.full(34, np.nan))

    def test_reference_is_periodic_and_in_joint_limits(self):
        env = ScubaEnv()
        np.testing.assert_allclose(reference(0), reference(2), atol=1e-12)
        for t in np.linspace(0, 2, 100):
            q = reference(t)
            self.assertTrue(np.all(q >= env.model.jnt_range[:, 0]))
            self.assertTrue(np.all(q <= env.model.jnt_range[:, 1]))

    def test_arms_clear_the_torso_throughout_the_dance(self):
        env = ScubaEnv()
        env.reset()
        for _ in range(600):
            env.step(env.scripted_action())
            for side in ("right", "left"):
                for limb in ("upper_arm", "forearm", "hand"):
                    arm = env.model.geom(f"{side}_{limb}").id
                    for bone in ("spine", "rib_1", "rib_2", "rib_3", "rib_4"):
                        torso = env.model.geom(bone).id
                        clearance = mujoco.mj_geomDistance(
                            env.model, env.data, arm, torso, 1.0, None)
                        self.assertGreater(clearance, 0.015,
                            f"{side}_{limb} intersects {bone}")

    def test_left_hand_sweeps_sideways_without_a_vertical_loop(self):
        env = ScubaEnv()
        env.reset()
        torso = env.model.body("torso").id
        hand = env.model.geom("left_hand").id
        positions = []
        for step in range(600):
            env.step(env.scripted_action())
            if step >= 100:  # Measure after the activation transient settles.
                rotation = env.data.xmat[torso].reshape(3, 3)
                positions.append(rotation.T @
                    (env.data.geom_xpos[hand] - env.data.xpos[torso]))
        depth, sideways, height = np.ptp(positions, axis=0)
        self.assertGreater(sideways, 0.35)
        self.assertLess(height, 0.06)
        self.assertLess(depth, 0.08)

    def test_torso_contacts_block_the_old_crossing_pose(self):
        env = ScubaEnv()
        env.reset()
        env.data.qpos[3:7] = [-0.90, 1.44, 0.0, 1.92]
        mujoco.mj_forward(env.model, env.data)
        arm_ids = {env.model.geom(n).id for n in
                   ("right_upper_arm", "right_forearm", "right_hand")}
        torso_ids = {env.model.geom(n).id for n in
                     ("spine", "rib_1", "rib_2", "rib_3", "rib_4")}
        self.assertTrue(any(c.dist < 0 and
            ((c.geom1 in arm_ids and c.geom2 in torso_ids) or
             (c.geom2 in arm_ids and c.geom1 in torso_ids))
            for c in env.data.contact))

    def test_muscles_track_for_multiple_cycles(self):
        env = ScubaEnv()
        q, targets, action, errors = rollout(env, 12)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all((action >= 0) & (action <= 1)))
        self.assertLess(np.sqrt(np.mean(errors)), 0.20)
        self.assertGreater(np.ptp(q[:, 8]), 0.5)
        self.assertGreater(np.ptp(q[:, 12]), 0.25)
        self.assertEqual(q.shape, targets.shape)


if __name__ == '__main__':
    unittest.main()
