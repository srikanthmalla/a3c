import gym
from gym import wrappers
from params import *
env = gym.make(environment)
if create_video:	
	env = wrappers.Monitor(env, './tmp',force=True)

print('observation space:',env.observation_space)
print('action space:',env.action_space)

observation = env.reset()
for _ in range(1000):
  # env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  # print(reward)
