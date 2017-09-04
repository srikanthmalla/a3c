import gym
from gym import wrappers
from params import *
from models import a3c
import numpy as np

model=a3c()
env = gym.make(environment)
if create_video:	
	env = wrappers.Monitor(env, './tmp',force=True)

print('observation space:',env.observation_space)
print('action space:',env.action_space)
print('action details',env.unwrapped.get_action_meanings())
with model.sess as sess:
	sess.run(model.init_op)
	for episode in range(1):
		print("episode:",episode)
		observation = env.reset()
		t = 0
		while True:
			env.render()
			action = env.action_space.sample() # your agent here (this takes random actions)
			observation=np.expand_dims(observation,axis=0)
			print(sess.run(model.p,feed_dict={model.observation:observation}))
			print(action)
			observation, reward, done, info = env.step(action)
			if done:
				print("Done after {} steps".format(t+1))
				break
