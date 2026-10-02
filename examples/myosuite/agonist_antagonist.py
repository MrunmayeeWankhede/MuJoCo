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
    action = env.action_space.sample()  #random muscle activations ("placeholder" for now, I will learn to control the muscles later)
    #as x climbs, the sine wave will oscillate between 0 and 1, which will cause the muscle activations to rise and fall
    #the 0.5 + 0.5 * ... is to shift the sine wave up so it is always positive, and to scale it so it is between 0 and 1
    #the 0.03 controls the speed of the oscillation, smaller is slower, larger is faster
    s = 0.5 + 0.5 * np.sin(x * 0.03)  #sine wave between 0 and 1, oscillating slowly
    action[0] = s  #activate the first muscle with the sine wave
    action[1] = 1 - s  #activate the second muscle with the inverse of the sine wave
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion
    
env.close()  #close the window and clean up


#WHAT I FOUND:
#muscle 0, 1, 2 = flexors (pull elbow down towards 0)
#muscle 3, 4, 5 = extensors (pull elbow up towards 2)
#relaxed resting angle = 0.593
#driving muscle 0 (flexor) and muslce 5 (extensor) 
#(action[5] = 1 - s) this makes them alternate like biceps and triceps

#this code swings the elbow through its full range of motion, by alternating the flexors and extensors
#THIS IS COOORDINATED MUSCLE CONTROL, which is what the nervous system does in real life