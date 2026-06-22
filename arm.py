#import packages
import time
import mujoco
import mujoco.viewer
import numpy as np

#this is the XML string for the Mujoco model
xml = """
<mujoco>
<worldbody> 
<light pos="0 0 3"/>
<geom type = "plane" size = "2 2 0.1"/>
<body pos = "0 0 1">
<joint name = "hinge1" type = "hinge" axis = "0 1 0" pos = "0 0 0"/>
<geom type = "capsule" fromto = "0 0 0 0.4 0 0" size = "0.04"/>
</body>
</worldbody>
<actuator>
<motor joint = "hinge1" gear = "1"/>
</actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
      

    



