from gym import wrappers
import numpy as np
from params import *
from models.dqn import *
import time
np.random.seed(1234)
model=dqn()
class dqn_agent():
	def __init__(self):
		self.episode=1
		self.mode=mode
		self.total_reward=0
		self.EPS=eps_start
		if self.mode=="test":
			model.load()
		self.env=gym.make(env_name)
		self.clear()
		self.render=render
	def clear(self):
		self.observations=[]
		self.new_observations=[]
		self.actions=[]
		self.Q=[]
		self.r=[]
	def getName(self):
		return self.__class__.__name__

	def train(self):
		model.train(self.observations,self.actions,self.Q,self.episode)

	def test(self):
		model.test(self.observations,self.actions,self.Q)
		self.clear()

	def run_episode(self):
		observation = self.env.reset()
		t = 0   #to calculate time steps
		while True:
			if (self.render):
				self.env.render('rgb_array')#show the game, xlib error with multiple thread
			action=self.predict_action([observation])
			self.observations.append(observation)
			self.actions.append([action])
			observation_new, reward, done, info = self.env.step(action)
			self.new_observations.append(observation_new)
			self.total_reward+=reward   
			self.r.append(reward) 
			t=t+1
			observation=observation_new
			if self.mode=="test":
				self.test()
			if done:
				self.Q_terminal=0
				#self.bellman_update() #can be used for batch
				print(" episode:",self.episode,"eps:","{0:.2f}".format(self.EPS)," reward:",self.total_reward," took {} steps".format(t))
				if self.mode=="train":
					self.q_update()
					self.train()
					model.log_details(self.total_reward,self.observations,self.actions,self.Q,self.episode,self.getName())
					self.total_reward=0
				break

	def run(self):
		start = time.time()
		while self.episode<=max_no_episodes:
			self.run_episode()
			self.clear()
			self.episode+=1
			self.EPS-=d_eps
			if self.episode%ckpt_episode==0:
				model.save(self.episode)
				print("saved model at episode {}".format(self.episode))
		end = time.time()
		print("took:",end - start)
	def test(self):
		start = time.time()
		model.load()
		self.run_episode()
		end = time.time()
		print("took:",end - start)	
	def q_update(self):
		self.Q=[]
		for i in range(len(self.r),0,-1):
			t=self.r[i-1]+self.Q_terminal
			self.Q.append([t])
			#self.R_terminal=t #normal bellman update
			self.Q_terminal=model.predict_Q([self.observations[i-1]],[self.actions[i-1]]) #batch bootstrap
		self.Q=np.flip(self.Q,axis=0)   
	
	def predict_action(self,observation):
		#here we use epsilon greedy exploration by tossing a coin
		actions=[[0],[1]]
		observations=np.tile(observation, (2, 1))
		q=model.predict_Q(observations,actions)
		action=actions[np.argmax(q)][0]
		# print(action)
		if np.random.uniform() < self.EPS:
			action=self.env.action_space.sample()
		return action

if __name__=="__main__":
	agent=dqn_agent()
	agent.run()
