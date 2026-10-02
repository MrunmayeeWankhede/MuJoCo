#import the packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time 

env = gym.make("myoArmReachRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers
env.reset()  #reset the model to its starting state

print("Number of muscles:", env.action_space.shape)
print("Number of joints:", core.mj_data.qpos.shape)

for x in range(1000):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    action = env.action_space.sample()  #random muscle activations 
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up