"""import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load a pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0", render_mode="human")  #this is a pre-built model, with a target to reach, and a reward function that scores how well I do

env.reset()  #reset the model to its starting state
for x in range(1000):  #step through 1000 frames of time
    env.render()  #draw the current state in a window
    action = env.action_space.sample()  #random muscle activations ("placeholder" for now, I will learn to control the muscles later)
    env.step(action)  #apply the random muscle activations to the model, and advance the physics
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up"""

#okay so the previous chunk doesnt work because 
#the hym render() path doesnt work for these environments
#instead of using env.render() i need to use the mujoco simulation directly
#like the pendulum_rhythm.py example, where i use mujoco.viewer.launch_passive(model, data) to open a window and run the simulation

"""import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import mujoco  #core physics engine
import mujoco.viewer  #interactive 3d viewer
import time

#laod the pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0")
env.reset()  #reset the model to its starting state

#look at what is inside the environment
#print attributes
core = env.unwrapped  #this is the core environment, without the gym wrappers
print([x for x in dir(core) if not x.startswith("_")])  #print all the attributes of the core environment, without the private ones that start with "_"

#reach past the gym wrappers to the real mujoco model and data
sim = env.unwrapped.sim  #this is the mujoco simulation object, which has the model and data
model = sim.model._model  #underlying MjModel
data = sim.data._data  #underlying MjData

#use the same viewer as the pendulum_rhythm.py example to open a window and run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:  #opens window
    while viewer.is_running():  #keep looping till window is closed
        action = env.action_space.sample()  #random muscle activations, like the data.ctrl[0] = 0.3 * np.sin(1.0 * data.time) in pendulum_rhythm.py
        env.step(action)  #like mujoco.mj_step(model, data) in pendulum_rhythm.py, this applies the random muscle activations to the model and advances the physics
        viewer.sync()  #redraw the window to show the new position
        time.sleep(0.01)  #pause briefly so I can watch the motion"""

#so this version of myosuite uses mujoco objects durectly as mj_model and mj_data
#mujoco objects not buried in sim, and thats why .sim fails
#just use mj_render()

"""
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#laod the pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0")
env.reset()  #reset the model to its starting state

#look at what is inside the environment
#print attributes
core = env.unwrapped  #this is the core environment, without the gym wrappers
#print([x for x in dir(core) if not x.startswith("_")])  #print all the attributes of the core environment, without the private ones that start with "_"


#reach past the gym wrappers to the real mujoco model and data
sim = env.unwrapped.sim  #this is the mujoco simulation object, which has the model and data
model = sim.model._model  #underlying MjModel
data = sim.data._data  #underlying MjData


for x in range(1000):  #step through 1000 frames of time
    core.mj_render()  #draw the current state in a window
    action = env.action_space.sample()  #random muscle activations ("placeholder" for now, I will learn to control the muscles later)
    print(action)  #print the random muscle activations
    env.step(action)  #apply the random muscle activations to the model, and advance the physics
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up
"""

#main learning - the action number is the number of muscles in the model, and the action values are the activations for each muscle, which are between 0 and 1
#numbers show muscle contractions (0 = relaxed, 1 = fully activation)
#instead of commanding a joint here, i am commanding a muscle
#then the muscles move the joint, and the physics engine simulates the motion
#THIS IS NEUROMECHANICSS woohoooo

#now comand the muslces deliberately, and see how the joint moves

import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time
import numpy as np

#laod the pre-built model - an elbow with muscles, tendons, and a target to reach
env = gym.make("myoElbowPose1D6MRandom-v0")
env.reset()  #reset the model to its starting state

#look at what is inside the environment
#print attributes
core = env.unwrapped  #this is the core environment, without the gym wrappers
#print([x for x in dir(core) if not x.startswith("_")])  #print all the attributes of the core environment, without the private ones that start with "_"
#action = np.zeros(env.action_space.shape)  #fully relaxed
#action[0] = 1.0  #fully activate the first muscle, which is the biceps
"""
#reach past the gym wrappers to the real mujoco model and data
sim = env.unwrapped.sim  #this is the mujoco simulation object, which has the model and data
model = sim.model._model  #underlying MjModel
data = sim.data._data  #underlying MjData
"""

for x in range(1000):  #step through 1000 frames of time
    core.mj_render()  #draw the current state in a window
    action = np.zeros(env.action_space.shape)  
    action[0] = 0.5 + 0.5*np.sin(x * 0.05)  #activation rises and falls forever so i dont miss it
    env.step(action)  #apply the random muscle activations to the model, and advance the physics
    print(round(core.mj_data.qpos[0], 3))  #print the current joint angle, rounded to 3 decimal places
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up