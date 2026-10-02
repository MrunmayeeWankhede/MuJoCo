"""Scuba on the official MyoSim full-body muscle model (pelvis supported)."""
from pathlib import Path
import argparse
import json
import time
import os

# These small dense control solves run faster with one BLAS thread on macOS.
# Set this before importing MuJoCo/NumPy (MuJoCo also imports NumPy).
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
from scipy.optimize import least_squares
from myo_sim.build.compose import build_spec, MODEL_REGISTRY

ROOT = Path(__file__).resolve().parent


class MyoScuba:
    def __init__(self):
        # Compose upstream anatomy in memory; only the root free joint is removed.
        # myo_sim 0.2.3 updates registry kwargs in place; restore its defaults.
        registration = MODEL_REGISTRY['myofullbody']
        original_kwargs = registration.build_kwargs.copy()
        try:
            spec = build_spec('myofullbody',
                build_kwargs={'add_root_freejoint': False}, inertia_floor=1e-4)
        finally:
            registration.build_kwargs.clear()
            registration.build_kwargs.update(original_kwargs)
        self.model = spec.compile()
        self.model.opt.timestep = 0.001
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)
        self.dt = 0.01
        self.qadr = {self.model.joint(i).name: int(self.model.jnt_qposadr[i])
                     for i in range(self.model.njnt)}
        self.equalities = []
        for i in range(self.model.neq):
            if self.model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                raise ValueError('Expected joint constraints in the MyoSim model')
            a, b = self.model.eq_obj1id[i], self.model.eq_obj2id[i]
            self.equalities.append((self.model.jnt_qposadr[a],
                self.model.jnt_qposadr[b] if b >= 0 else -1, self.model.eq_data[i, :5].copy()))
        dependent = {a for a, _, _ in self.equalities}
        self.independent = np.array([i for i in range(self.model.nq) if i not in dependent])
        self.weights = np.array([3.0 if any(self.model.joint(i).name.startswith(n)
            for n in ('elv_angle_', 'shoulder_elv_', 'shoulder_rot_', 'elbow_flexion_',
                      'pro_sup_', 'deviation_', 'flexion_'))
            else 1.0 for i in self.independent])
        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.moments = np.zeros((self.model.nu, self.model.nv))
        self.activation = np.zeros(self.model.nu)
        # The mirrored left arm flips its local palmar side. Read that side
        # from anatomical flexor/extensor attachments rather than assuming -Z.
        self.palm_local_sign = {}
        for side in ('r', 'l'):
            flexor_z = np.mean([self.model.site(name+side).pos[2]
                               for name in ('FDS3-P4_', 'FDP3-P4_')])
            extensor_z = self.model.site('EDC3-P4_'+side).pos[2]
            self.palm_local_sign[side] = float(np.sign(flexor_z-extensor_z))
        self.base = self.model.qpos0.copy()
        self.prepare_motion()
        self.reset()

    def constrain(self, q):
        """Apply upstream polynomial joint couplings to a reference pose."""
        q = q.copy()
        for a, b, coeff in self.equalities:
            q[a] = np.polynomial.polynomial.polyval(q[b] if b >= 0 else 0, coeff)
        return q

    def projection(self, q):
        """Map independent joint velocities to the full anatomical chain."""
        p = np.eye(self.model.nv)[:, self.independent]
        for a, b, coeff in self.equalities:
            p[a] = 0 if b < 0 else np.polynomial.polynomial.polyval(q[b],
                   np.arange(1, 5)*coeff[1:])*p[b]
        return p

    def hand_position(self, data, side):
        return data.body('thirdmc_'+side).xpos.copy()

    def palm_normal(self, data, side):
        """World palmar normal, accounting for the mirrored left anatomy."""
        return (self.palm_local_sign[side]
                * data.body('thirdmc_'+side).xmat.reshape(3, 3)[:, 2])

    def right_palm_normal(self, data):
        return self.palm_normal(data, 'r')

    def left_palm_alignment(self, data):
        # The chest is behind the sweeping hand at +Y in model coordinates.
        return float(self.palm_normal(data, 'l') @ np.array([0, 1, 0]))

    def right_palm_alignment(self, data):
        nose = data.body('head').xpos + np.array([0, -.09, -.035])
        toward_nose = nose-self.hand_position(data, 'r')
        toward_nose /= max(np.linalg.norm(toward_nose), 1e-8)
        return float(self.right_palm_normal(data) @ toward_nose)

    def solve_arm(self, side, point, initial):
        """IK supplies targets only; the physical data never follows IK directly."""
        names = ['elv_angle_', 'shoulder_elv_', 'shoulder_rot_', 'elbow_flexion_', 'pro_sup_']
        names += ['deviation_', 'flexion_']
        ids = [self.qadr[n+side] for n in names]
        ranges = np.array([self.model.joint(n+side).range for n in names])
        arm_geoms = [self.model.geom(n+side).id for n in
                     ('humerus_coll_', 'ulna_coll_', 'radius_coll_', '3mcskin_coll_')]
        chest_geoms = [self.model.geom(n).id for n in
                       ('thorax_coll1', 'thorax_coll2', 'thorax_coll3')]
        def residual(x):
            q = self.base.copy()
            q[ids] = x
            self.kin.qpos[:] = self.constrain(q)
            mujoco.mj_forward(self.model, self.kin)
            clearance = [max((0.0 if side == 'r' else .008)-mujoco.mj_geomDistance(self.model, self.kin,
                         arm, chest, .1, None), 0)
                         for arm in arm_geoms for chest in chest_geoms]
            elbow_preference = []
            palm_preference = []
            if side == 'r':
                # A nose gesture bends the forearm upward with the elbow below
                # the hand, rather than reaching over the top of the skull.
                elbow = self.kin.body('ulna_r').xpos
                elbow_target = point + np.array([-.18, .015, -.24])
                elbow_preference = 5*(elbow-elbow_target)
                nose = self.kin.body('head').xpos + np.array([0, -.09, -.035])
                toward_nose = nose-self.hand_position(self.kin, 'r')
                toward_nose /= max(np.linalg.norm(toward_nose), 1e-8)
                palm_preference = self.right_palm_normal(self.kin)-toward_nose
            else:
                # The sweeping palm stays toward the chest instead of spinning
                # with the shoulder. Wrist/forearm motion compensates the sweep.
                palm_preference = self.palm_normal(self.kin, 'l')-np.array([0, 1, 0])
            return np.r_[8*(self.hand_position(self.kin, side)-point),
                         20*np.asarray(clearance), elbow_preference, palm_preference, .025*(x-initial)]
        answer = least_squares(residual, initial, bounds=(ranges[:,0]+1e-5, ranges[:,1]-1e-5),
                               max_nfev=100, ftol=1e-8, xtol=1e-8)
        return ids, answer.x

    def prepare_motion(self):
        self.kin.qpos[:] = self.constrain(self.base)
        mujoco.mj_forward(self.model, self.kin)
        head = self.kin.body('head').xpos.copy()
        # In the library's world coordinates: X is lateral, forward is -Y, Z is up.
        right_point = head + np.array([-0.045, -0.15, -0.06])
        self.right_ids, right = self.solve_arm('r', right_point, np.array([1.2,1.2,0,2.0,0,0,0]))
        self.base[self.right_ids] = right
        # Left hand sweeps across the front at a constant requested height/depth.
        self.left_points = np.array([[x, head[1]-.46, head[2]-.28]
                                    for x in np.linspace(-.10, .38, 33)])
        poses = []
        guess = np.array([1.5,1.2,0,.4,0,0,0])
        for point in self.left_points:
            self.left_ids, guess = self.solve_arm('l', point, guess)
            poses.append(guess.copy())
        self.left_poses = np.array(poses)

    def target(self, t):
        phase = 2*np.pi*.4*t
        s = np.sin(phase)
        q = self.base.copy()
        u = (s+1)/2*(len(self.left_poses)-1)
        i = min(int(u), len(self.left_poses)-2)
        q[self.left_ids] = (1-(u-i))*self.left_poses[i]+(u-i)*self.left_poses[i+1]
        for side, sign in [('r',1),('l',-1)]:
            q[self.qadr['hip_flexion_'+side]] = .16 + sign*.12*s
            q[self.qadr['knee_angle_'+side]] = .30 + sign*.20*s
            q[self.qadr['ankle_angle_'+side]] = -.05
        q[self.qadr['lat_bending']] = .025*s
        q[self.qadr['axial_rotation']] = .025*s
        return self.constrain(q)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.target(0)
        self.activation[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def controller(self):
        """Allocate computed generalized torque to bounded muscle excitations."""
        m, d = self.model, self.data
        eps = .002
        target = self.target(d.time)
        vel = (self.target(d.time+eps)-self.target(d.time-eps))/(2*eps)
        accel = 225*(target-d.qpos) + 30*(vel-d.qvel)
        mujoco.mj_fullM(m, self.mass, d.qM)
        self.moments.fill(0)
        for a in range(m.nu):
            start, count = d.moment_rowadr[a], d.moment_rownnz[a]
            self.moments[a, d.moment_colind[start:start+count]] = d.actuator_moment[start:start+count]
        gain = np.array([mujoco.mju_muscleGain(d.actuator_length[a],d.actuator_velocity[a],
                   m.actuator_lengthrange[a],m.actuator_acc0[a],m.actuator_gainprm[a,:9]) for a in range(m.nu)])
        bias = np.array([mujoco.mju_muscleBias(d.actuator_length[a],m.actuator_lengthrange[a],
                   m.actuator_acc0[a],m.actuator_biasprm[a,:9]) for a in range(m.nu)])
        projection = self.projection(d.qpos)
        desired = projection.T @ (self.mass@accel + d.qfrc_bias-d.qfrc_passive-self.moments.T@bias)
        mapping = (projection.T@self.moments.T)*gain[None,:]
        # Normalize generalized torques, then bounded muscle allocation via projected gradient.
        scale = np.maximum(np.linalg.norm(mapping,axis=1), .5)
        b = mapping/scale[:,None]*self.weights[:,None]
        y = desired/scale*self.weights
        lipschitz = max(float(np.linalg.eigvalsh(b@b.T)[-1]) + .002, 1)
        a = self.activation.copy()
        v = a.copy()
        momentum = 1.0
        for _ in range(80):
            nxt = np.clip(v-(b.T@(b@v-y)+.002*v)/lipschitz,0,1)
            new_momentum = (1+np.sqrt(1+4*momentum*momentum))/2
            v = nxt+(momentum-1)/new_momentum*(nxt-a)
            a = nxt
            momentum = new_momentum
        self.activation[:] = a
        return a

    def step(self):
        self.data.ctrl[:] = self.controller()
        mujoco.mj_step(self.model,self.data,nstep=round(self.dt/self.model.opt.timestep))
        if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel))>100:
            raise RuntimeError('MyoSim simulation became unstable')
        return float(np.mean((self.data.qpos[self.independent]-self.target(self.data.time)[self.independent])**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--hide-muscles', action='store_true',
                        help='Hide tendon lines in the viewer to see bones clearly')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    sim = MyoScuba()
    errors, positions, controls, hand_positions, palm_alignment, left_palm_alignment = [], [], [], [], [], []

    def run(viewer=None):
        for _ in range(int(np.ceil(args.seconds/sim.dt))):
            start = time.monotonic()
            errors.append(sim.step())
            positions.append(sim.data.qpos.copy())
            controls.append(sim.data.ctrl.copy())
            hand_positions.append(np.r_[sim.hand_position(sim.data, 'r'),
                                        sim.hand_position(sim.data, 'l')])
            palm_alignment.append(sim.right_palm_alignment(sim.data))
            left_palm_alignment.append(sim.left_palm_alignment(sim.data))
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, sim.dt-(time.monotonic()-start)))

    if args.headless:
        run()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.azimuth = -90
            viewer.cam.elevation = -5
            viewer.cam.distance = 3
            viewer.cam.lookat[:] = [0, .1, 1]
            if args.hide_muscles:
                viewer.opt.tendongroup[:] = 0
            run(viewer)
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'myo_scuba_rollout.npz', qpos=positions,
        excitation=controls, hand_positions=hand_positions, dt=sim.dt,
        joint_names=np.array(list(sim.qadr)))
    result = {
        'model': 'myofullbody',
        'pelvis_supported': True,
        'muscles': sim.model.nu,
        'steps': len(errors),
        'tracking_rmse_rad': float(np.sqrt(np.mean(errors))),
    }
    if len(hand_positions) > round(1/sim.dt):
        movement = np.ptp(np.asarray(hand_positions)[round(1/sim.dt):], axis=0)
        result['left_hand_lateral_range_m'] = float(movement[3])
        result['left_hand_vertical_range_m'] = float(movement[5])
        result['right_palm_min_alignment_cosine'] = float(
            min(palm_alignment[round(1/sim.dt):]))
        result['left_palm_min_alignment_cosine'] = float(
            min(left_palm_alignment[round(1/sim.dt):]))
    (output/'myo_scuba_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
