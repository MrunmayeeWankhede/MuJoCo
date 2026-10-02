"""Integration checks for the optional anatomical MyoSim workflow."""
import importlib.util
import unittest
import numpy as np
import mujoco


@unittest.skipUnless(importlib.util.find_spec('myo_sim'), 'Install requirements-myo.txt')
class MyoScubaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from myo_scuba import MyoScuba
        cls.sim = MyoScuba()

    def test_official_muscles_and_joint_constraints(self):
        sim = self.sim
        self.assertEqual(sim.model.nu, 416)
        self.assertEqual(sim.model.nq, 122)
        self.assertTrue(np.all(sim.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE))
        self.assertTrue(np.all(sim.model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON))
        self.assertGreater(sim.model.npair, 0)
        for t in np.linspace(0, 2.5, 30):
            q = sim.target(t)
            for a, b, coeff in sim.equalities:
                expected = np.polynomial.polynomial.polyval(q[b] if b >= 0 else 0, coeff)
                self.assertAlmostEqual(q[a], expected, places=10)
            for j in sim.independent:
                if sim.model.jnt_limited[j]:
                    self.assertGreaterEqual(q[j], sim.model.jnt_range[j, 0]-1e-7)
                    self.assertLessEqual(q[j], sim.model.jnt_range[j, 1]+1e-7)

    def test_mirrored_left_palm_faces_inward_in_reference_motion(self):
        sim = self.sim
        # Mirroring flips the flexor/extensor sides: a shared -Z normal would
        # incorrectly identify the back of the left hand as its palm.
        self.assertEqual(sim.palm_local_sign, {'r': -1.0, 'l': 1.0})
        for t in np.linspace(0, 2.5, 100):
            sim.kin.qpos[:] = sim.target(t)
            mujoco.mj_forward(sim.model, sim.kin)
            self.assertGreater(sim.left_palm_alignment(sim.kin), 0.95)

    def test_physics_drives_the_scuba_with_muscle_excitation(self):
        sim = self.sim
        sim.reset()
        positions, knees, errors, contacts, palm_alignment, left_palm_alignment = [], [], [], [], [], []
        for step in range(1200):
            errors.append(sim.step())
            self.assertTrue(np.isfinite(sim.data.qpos).all())
            self.assertTrue(np.all((sim.data.ctrl >= 0) & (sim.data.ctrl <= 1)))
            self.assertTrue(np.all(sim.data.qfrc_applied == 0))
            self.assertTrue(np.all(sim.data.xfrc_applied == 0))
            if step >= 100:
                positions.append(np.r_[sim.hand_position(sim.data, 'r'),
                                       sim.hand_position(sim.data, 'l')])
                knees.append(sim.data.qpos[sim.qadr['knee_angle_r']])
                palm_alignment.append(sim.right_palm_alignment(sim.data))
                left_palm_alignment.append(sim.left_palm_alignment(sim.data))
            for c in sim.data.contact:
                names = sim.model.geom(c.geom1).name+sim.model.geom(c.geom2).name
                if 'thorax_coll' in names:
                    contacts.append(c.dist)
        positions = np.array(positions)
        movement = np.ptp(positions, axis=0)
        self.assertLess(np.sqrt(np.mean(errors)), 0.3)
        self.assertGreater(movement[3], 0.35)  # Side-to-side hand sweep.
        self.assertLess(movement[5], 0.06)
        self.assertLess(np.linalg.norm(movement[:3]), 0.10)  # Raised hand stays near face.
        self.assertGreater(min(left_palm_alignment), 0.85)  # Sweeping palm stays inward.
        self.assertGreater(min(palm_alignment), 0.85)  # Palm faces the nose throughout.
        self.assertGreater(np.ptp(knees), 0.25)
        self.assertGreater(min(contacts, default=0), -0.005)


if __name__ == '__main__':
    unittest.main()
