#import the packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

env = gym.make("myoHandPoseRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers
env.reset()  #reset the model to its starting state

for x in range(1000):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    action = env.action_space.sample()  #random muscle activations 
    s = 0.5 + 0.5*np.sin(x*0.015)  #oscillate between 0 and 1
    action[2] = s  #blue muscle = pulls joint one way
    action[21] = 1 - s  #fred muscle = pulls joint opposite way
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up

#HAND GESTURE: driving an opposing muscle pair on the full hand.
#picked muscle 2 (blue on joint 0) vs muscle 21 (red on joint 0) from the heatmap.
#result: the PINKY curls and uncurls rhythmically, with the whole hand.
#shaking a little because these muscles also move other joints (coupling) can be called noise?.
#manually picking the pairs gives correct but noisy motion.
#clean multi-joint control by hand isn't really possible 
#this is why RL is needed for complex coordinated movement.
