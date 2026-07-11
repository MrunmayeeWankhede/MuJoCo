#prints the anatomical name of every joint in the myoArmReachRandom-v0 environment

import myosuite
import gymnasium as gym

env = gym.make("myoArmReachRandom-v0")
core = env.unwrapped
model = core.mj_model

for i in range(model.njnt):
    name = model.joint(i).name
    print(i, name)

#for this model myoArmReachRandom-v0, the joint names are:
#joints 0-9 = shoulder girdle (clavicle/scapula) 
#joint 11 = shoulder_elv (raise the arm)
#joint 13 = shoulder_rot (rotate the shoulder)
#joint 14 = elbow_flexion (bend the elbow)
#joint 15 = pro_sup (forearm rotation (palm up/down))
#joints 10/12 = coupled shoulder mechanism (mirror each other)
