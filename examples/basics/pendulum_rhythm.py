#import packages
import time  #time.sleep() to slow playback
import mujoco  #core physics engine
import mujoco.viewer  #interactive 3d viewer
import numpy as np  

#this is the XML string for the Mujoco model
#this defines the physical body- a rod on a hinge which i can push
xml = """
<mujoco>
<worldbody> 
<light pos="0 0 3"/>  <!--light so scene isnt dark-->
<geom type = "plane" size = "2 2 0.1"/>  <!--ground plane, foxed, no body needed-->
<body pos = "0 0 1">  <!--rod's body, plaved 1 unit above the ground plane-->
<joint name = "hinge1" type = "hinge" axis = "0 1 0" pos = "0 0 0" damping="0.2"/>  <!--hinge joint, rotates about y axis, at the base of the rod-->
<geom type = "capsule" fromto = "0 0 0 0.4 0 0" size = "0.04"/>  <!--rod geometry, a capsule from the hinge to 0.4 units along the x axis, radius 0.04-->
</body>
</worldbody>
<actuator>
<motor joint = "hinge1" gear = "1"/>
</actuator>
</mujoco>
"""

#build the simulation
model = mujoco.MjModel.from_xml_string(xml)  #parse XML into fixed blueprint (MjModel)
data = mujoco.MjData(model)  #create the live dynamic state (MjData) from the blueprint


#run loop
with mujoco.viewer.launch_passive(model, data) as viewer:  #opens window, I run the physics
    while viewer.is_running():  #keep looping until window is closed
        data.ctrl[0] = 0.3 * np.sin(1.0 * data.time)  #this is the command - a smooth push that swings between -3 and 3, the sine wave is a function of time, kinda like amplitude knob
        mujoco.mj_step(model, data)  #advance the physics by 1 timestep, using the command
        print(data.qpos[0])  #print the rod's current joint angle
        viewer.sync()  #redraw the window to show the new position
        time.sleep(0.01)  #pause briefly so i can watch the motion
      

    
#WHAT I LEARNED:

#MuJoCo is only the body (physics engine)
#My Python code is the brain (control engine)
#each step it reads state and writes a command into data.ctrl
#a constant command just holds a position and lets gravity do its thing
#a command that chnages over time is ehat makes ME the source of motion, not gravity

#AMPLITUDE
#this is the number multiplying thr sine (n * np.sin(...))
#this is HOW HARD I PUSH the rod
#the bigger the number, the more torque I apply to the hinge, and the higher the rod swings
#if this is too weak and gravity/damping win, the rod barely moves
#if the number is too big, the rod swings so high it hits the ceiling and the simulation explodes
#if it is just strong enough, the rod swings up and down in a nice smooth motion

#FREQUENCY
#this is the number inside the sine (np.sin(n * data.time))
#this is HOW FAST I PUSH the rod
#the bigger the number, the faster I push the rod back and forth
#if this is too slow, the rod swings up and down in a lazy motion
#if this is too fast, the rod swings so quickly it hits the ceiling and the simulation explodes
#key finding ehich i found surprising:-
#higher frequency = smaller swing. the rod has mass so it cant keep up with a fast fush
#fast drive = tiny atcg, slow drive = big, lazy arc
#somewhere in between is the "resonant frequency" where the rod swings the highest (sweet spot)

#DAMPING
#this is the number in the hinge joint (damping="n")
#this is the FRICTION in the hinge joint, which resists motion
#it lets energy LEAK OUT of the system
#key finding:-
#with NO damping, the rod is frictionless and swings forever
#energy just sloshes between height and speed and never escapes, and never settles
#but real joints also have some damping (coz friction lol)
#tradeoff: too much damping = rod barely moves/creeps to a resting spot
#too little damping = rod spins over the tip
#a clean back and forth swing is somewhere in between, where the rod swings up and down and eventually settles to a resting position

#DRIVE vs DAMPING
#this is the relationship between the command (drive) and the hinge friction (damping)
#my sine push PUMPS energy in, and the damping DRAINS the energy out
#when they balance, the rod settles into a steady rhythm
#this is like a toy version of a "central pattern generator" in the brain
#this is a neural circuit that produces rhythmic motion like walking or swimming

#the data.qpos[0] is the current hinge angle of the rod, which is a function of the command, damping, and gravity
#this is used to measure the rod's motion, and can be used to tune the command to get the desired motion

#data.time is MuJoCo's internal clock, which is used to drive the sine wave command
#this keeps the rhythm correcr and synced to real simulation time

#WHY SINE WAVE?
#a sine wave is the simplest and smoothest repeating back and forth signal
#it rises, falls, goes -ve, comes back up, and repeats forever
#this is the shape i need to drive rhythmic motion, and is the basis of many natural oscillations in biology and physics


