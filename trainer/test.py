from pz_battlesnake.env import standard_v0

env = standard_v0.env()
num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).num_actions()



