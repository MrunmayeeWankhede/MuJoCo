"""
RANDOM MUSCLE ACTIVATION IN MYO HAND MODEL

#import packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the full hand/forearm model 
#29 bones, 23 joints, 39 muscles
env = gym.make("myoHandPoseRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers

env.reset()  #reset the model to its starting state

print("Number of muscles:", env.action_space.shape)

for x in range(1000):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    action = env.action_space.sample()  #random muscle activations 
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up
"""

"""
COMMANDING A SINGLE MUSCLE IN MYO HAND MODEL

#import packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the full hand/forearm model 
#29 bones, 23 joints, 39 muscles
env = gym.make("myoHandPoseRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers

env.reset()  #reset the model to its starting state

print("Number of muscles:", env.action_space.shape)

for x in range(1000):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    action = np.zeros(env.action_space.shape)  #ALL 39 MUSCLES OFF
    action[0] = 1.0  #fully activate the first muscle
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion

env.close()  #close the window and clean up
"""

"""
PRINTING THE JOINTS THAT MOVE MOST WHEN A MUSCLE IS ACTIVATED

#import packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the full hand/forearm model 
#29 bones, 23 joints, 39 muscles
env = gym.make("myoHandPoseRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers

env.reset()  #reset the model to its starting state
before = core.mj_data.qpos.copy()  #save the initial joint angles

action = np.zeros(env.action_space.shape)  #ALL 39 MUSCLES OFF
action[5] = 1.0  #fully activate the first muscle

for x in range(300):  #step through 1000 frames of time
    core.mj_render() #draw the current state in a window
    env.step(action)  #apply the random muscle activations to the model, and advance
    time.sleep(0.01)  #pause briefly so I can watch the motion

after = core.mj_data.qpos.copy()  #save the final joint angles
change = after - before  #calculate the change in joint angles

#show the joints that moved the most
print("Biggest movers:", np.argsort(np.abs(change))[-3:])  #print the indices of the 3 joints that moved the most
print("Amount of change:", np.round(change[np.argsort(np.abs(change))[-3:]], 3))  #print the amount of change for those joints

env.close()  #close the window and clean up
"""

#MAKING A MAP OF MUSCLES TO THE JOINTS THEY MOVE MOST

#import packages
import numpy as np
import myosuite  #muscoskeletal library (built on mujoco)
import gymnasium as gym  #environment library myosuite uses
import time

#load the full hand/forearm model 
#29 bones, 23 joints, 39 muscles
env = gym.make("myoHandPoseRandom-v0")
core = env.unwrapped  #this is the core environment, without the gym wrappers

#dictionary to hold the muscle index and the list of joints it moves most
muscle_map = {}

n_muscles = env.action_space.shape[0]  #number of muscles in the model (39)

for m in range(n_muscles):  #for each muscle in the model
    env.reset()  #reset the model to its starting state
    before = core.mj_data.qpos.copy()  #save the initial joint angles

    action = np.zeros(env.action_space.shape)  #ALL 39 MUSCLES OFF
    action[m] = 1.0  #fully activate only the current muscle m

    for x in range(200):  #step through 1000 frames of time
        env.step(action)  #apply the random muscle activations to the model, and advance

    after = core.mj_data.qpos.copy()  #save the final joint angles
    change = after - before  #calculate the change in joint angles

    top_joints = np.argsort(np.abs(change))[-3:]  #get the indices of the 3 joints that moved the most
    muscle_map[m] = [int(j) for j in top_joints]  #store the muscle index and the list of

    print("Muscle", m, "moves joints", list(top_joints))

env.close()  #close the window and clean up

#print the final muscle map
print("FULL MUSCLE MAP")
for m, joints in muscle_map.items():
    print("Muscle", m, "moves joints", joints)


#MUSCLE to JOINT MAP for MyoHand (39 muscles, 23 joints)
#activates each muscle alone, records the 3 joints it moves most.
#findings:
#every muscle moves around 3 joints (muscles are coupled)
#joint 0 shows up in tons of muscles (probably a big central joint (wrist?))
#this records which joints move, not direction (flex/extend) or strength