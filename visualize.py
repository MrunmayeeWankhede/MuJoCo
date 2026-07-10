#import the packages
import matplotlib
matplotlib.use("Agg")     #save to file instead of opening a window
import matplotlib.pyplot as plt

import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses

import os

env = gym.make("myoHandPoseRandom-v0")  #load the full hand/forearm model
core = env.unwrapped  #this is the core environment, without the gym wrappers
n_muscles = env.action_space.shape[0]  #get the number of muscles in the model
n_joints = core.mj_data.qpos.shape[0]  #get the number of joints in the model

#build a grid: rows = muscles, columns = joints, value = jpw much it moved
grid = np.zeros((n_muscles, n_joints))  #initialize a grid of zeros

for m in range(n_muscles):  #for each muscle
    env.reset()  #resey yhe model to its starting state
    before = core.mj_data.qpos.copy()  #save the initial joint angles
    action = np.zeros(env.action_space.shape)  #initialize an action of all zeros (all muscles off)
    action[m] = 1.0  #fully activate the current muscle

    for x in range(200):  #step through 200 frames of time
        env.step(action)  #apply the muscle activation to the model, and advance the physics
    after = core.mj_data.qpos.copy()  #save the final joint angles
    grid[m] = after - before  #signed change for every joint (not just top 3)

env.close()  #close the window 

#make the results folder
os.makedirs("results", exist_ok=True)

#draw the heatmap of the grid
plt.figure(figsize=(10, 8))
plt.imshow(grid, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)  #set vmin and vmax to -2 and 2 for better color contrast
plt.colorbar(label="Joint Movement (radians)") 
plt.xlabel("Joint Index")  
plt.ylabel("Muscle Index")  
plt.title("MyoHand: whcich muscles move which joints + which direction")
plt.tight_layout()

#save the heatmap to a file
plt.savefig("results/muscle_joint_heatmap.png", dpi=150)
print("Heatmap saved to results/muscle_joint_heatmap.png")
