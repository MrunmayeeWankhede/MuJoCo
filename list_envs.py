#import packages
import myosuite
import gymnasium as gym

#list every environment in myosuite registered
myo_envs = [env for env in gym.envs.registry.keys() if "myo" in env.lower()]

for name in myo_envs:
    print(name)  #print the name of the environment