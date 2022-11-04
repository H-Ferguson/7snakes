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

episodes = 1000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        actions = {}
        # Run agent on the state
        for agent in env.agents: 
          actions[agent] = snakes[agent].act(state[agent])

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(actions)
        # env.render(mode="human")
        
        # this is dumb but 
        if len(env.agents) <= 0:
          break

        # Remember
        for snake in snakes:
          this_snake = snakes[snake]
          snake_state = state[snake]
          snake_next_state = next_state[snake]
          snake_action = actions[snake]
          snake_reward = reward[snake]
          snake_done = done[snake]
          this_snake.cache(snake_state, snake_next_state, snake_action, snake_reward, snake_done)

          # Learn
          q, loss = this_snake.learn()

          # Logging
          logger.log_step(snake_reward, loss, q)
          logger.log_episode()
          if e % 20 == 0:
            logger.record(episode=e, epsilon=this_snake.exploration_rate, step=this_snake.curr_step)

        # Update state
        state = next_state

        

    

    