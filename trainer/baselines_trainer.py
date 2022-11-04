from pz_battlesnake.env import standard_v0


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

env = standard_v0.env()
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()