# Test Package Install Worked -- Headless Random Sampling of Custom Env


import gym
import time

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


# Read in Custom Env ".x86_64" for Linux & ".app" for MacOS
unity_env = UnityEnvironment("CustomUnityEnvironments/Environment-1.x86_64", no_graphics=True) #no_graphics=True to run in headless mode
env = UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=False)


# Print Env Shapes & Actual Actions/Observations
print(f"Action Space Shape: {env.action_space.shape} - {env.action_space.dtype}") #all actions are continuous but bounded between {-1 & 1} (i.e lower & upper bounds)
print(f"Observation Space Shape: {env.observation_space.shape} - {env.observation_space.dtype}")

print("Action Space:", env.action_space) #all actions are discrete movements
print("Observation Space:", env.observation_space) #Continuous Observations


# Test Random Sampling of Env
for i in range(0, 3):
    s, done = env.reset(), False



# Close Port
env.close()
