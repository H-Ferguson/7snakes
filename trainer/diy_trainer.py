from pz_battlesnake.env import standard_v0, solo_v0
from snakenet import SnakeNet
import torch
import torch.optim as optim
import numpy as np 
from logger import MetricLogger
from snake import Snake
from pathlib import Path
import datetime

env = standard_v0.env()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

action_dim = env.action_space(env.possible_agents[0]).n
# play against urself
snakes = {}
for agent in env.possible_agents:
  snake = Snake(state_dim=(1, 11, 11), action_dim=action_dim, save_dir=save_dir)
  snakes[agent] = snake

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()
    dones = {}
    for agent in env.agents:
      dones[agent] = False

    # Play the game!
    while True:

        actions = {}
        # Run agent on the state
        for agent in env.agents: 
          actions[agent] = snakes[agent].act(state[agent])

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(actions)
        env.render(mode="human")

        # Remember
        for snake in snakes:
          snakes[snake].cache(state[snake], next_state[snake], actions[snake], reward[snake], done[snake])

          # Learn
          q, loss = snakes[snake].learn()

          # Logging
          logger.log_step(reward[snake], loss, q)
          logger.log_episode()
          if e % 20 == 0:
            logger.record(episode=e, epsilon=snakes[snake].exploration_rate, step=snakes[snake].curr_step)

        # Update state
        state = next_state

        for snake in snakes:
          dones[snake] = done[snake]
        all_done = False 
        for snake in snakes: 
          if done[snake] == False:
            break;
          all_done = True

        # Check if end of game
        if all_done:
            break

    

    