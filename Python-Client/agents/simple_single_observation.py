from envs.single_observation import SingleObservation
import gym
env = SingleObservation()
for i_episode in range(20):
    env.reset()
    action = env.action_space.sample()
    for t in range(10000):
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
