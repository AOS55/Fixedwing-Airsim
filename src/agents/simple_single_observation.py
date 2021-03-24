from envs.single_observation import SingleObservation
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.utils.tensorboard import SummaryWriter
import wandb

wandb.init(project="SingleObserverTest")
config = wandb.config
config.dropout = 1.0

# writer = SummaryWriter()
env = SingleObservation()
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("CnnPolicy", env, action_noise=action_noise, verbose=1, buffer_size=10000,
            tensorboard_log="./td3_learning_tensorboard")
model.learn(total_timesteps=10)
wandb.watch(model)

obs = env.reset()
for _ in range(10):
    print("running")
    action, _states = model.predict(obs)
    # action = env.action_space.sample()
    for t in range(10000):
        obs, rewards, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            obs = env.reset()
            break

env.render()
# writer.flush()
env.close()
