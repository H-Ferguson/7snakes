"""
the pz_battlesnake library https://github.com/DaBultz/pz-battlesnake is kind of brittle
this file can be used as a sanity check/ debugger for each of the envs to make sure its responding to moves and 
state updates properly

Might eventually submit a PR for my "fixed" version of the lib 

instances an env and makes each agent pick a random move
"""
from pz_battlesnake.env import standard_v0

env = standard_v0.env() 

for _ in range(10):
    observations = env.reset()
    print(observations)
    env.render()
    done = False
    while not done:
        for agent in env.agents:
            action = env.action_space(agent).sample() if not termination else None
            observation, reward, termination, truncation, info = env.step(action)
            env.step(action)
        # env.render()
        done = not env.agents