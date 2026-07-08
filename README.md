# Me trying to learn MuJoCo so I can make cool simulations

<img width="396" height="352" alt="image" src="https://github.com/user-attachments/assets/ca3ca6ad-c2d5-4b1a-9111-9fedad383fc8" />


## My Goal
Make the skeleton hit **the scuba** :D

(yes, an actual musculoskeletal model doing the scuba. it's ambitious. it's fun. that's the point.)

## Where I'm at
- [x] Falling box: got the basic model → data → step loop working
- [x] Pendulum on a hinge: driving a joint with `data.ctrl`
- [x] Sine-wave control: making MY command create the motion, not gravity
- [x] Loaded a real MyoSuite elbow: drove an actual muscle to flex rhythmically using a pre-constructed model
- [x] Drive multiple muscles together (agonist/antagonist pairs)
- [ ] Load a bigger/full-body model
- [ ] Hand-author a coordinated motion (scuba attempt #1)
- [ ] Train it properly (RL/motion targets) for a real scuba

## What I've learned so far
- MuJoCo is just the **body** (physics). My Python is the **brain** that sends commands each step
- A constant command moves a limb once then holds. *Ongoing* motion needs a command that changes over time
- A "relaxed" limb still moves, because gravity and passive tendon tension are always acting
- Muscle activation (0-1) is the command for a real muscle, not raw torque
- When my eyes can't see the motion, `qpos` shows it in numbers

## Files
- `arm.py` - falling box + first pendulum
- `pendulum_rhythm.py` - sine-wave driven pendulum (with notes)
- `myo_model.py` - driving a real MyoSuite elbow muscle
