"""
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0")

core = env.unwrapped  #this is the core environment, without the gym wrappers

env.reset()  #reset the model to its starting state
action = np.zeros(env.action_space.shape)  #fully relaxed condition

#determine the final value of the angle set by muscle at index 0 (biceps?) 
#chaange index to see how the other muscles affect the joint angle (1, 2, 3, 4, 5)
action[5] = 1.0  #fully activate the first muscle, which is the biceps

for x in range(300):  #step through 300 frames of time
    core.mj_render() #draw the current state in a window
    env.step(action)  #apply the random muscle activations to the model, and advance the physics
    print(round(core.mj_data.qpos[0], 3))  #print the current angle 
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up
"""

import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0")

core = env.unwrapped  #this is the core environment, without the gym wrappers

env.reset()  #reset the model to its starting state

for x in range(1000):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    action = np.zeros(env.action_space.shape)  #fully relaxed condition
    action[0] = 1.0  #fully activate the flexor muscle
    action[5] = 1.0 #fully activate the extensor muscle
    env.step(action)  #apply the random muscle activations to the model, and advance
    print(round(core.mj_data.qpos[0], 3))  #print the current angle
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up


#WHAT I FOUND:
#cocontraction does not necessarily settle at the exact midpoint
#it settles wherever the stronger muscle drags it
#weaker muscle resists