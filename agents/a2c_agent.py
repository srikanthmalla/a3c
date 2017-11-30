from models.a2c import a2c
import numpy as np
import time
from params import *
from helper_funcs import *
from gym import wrappers
from threading import Thread
from gym.envs.classic_control import rendering
#to get same random for rerun
np.random.seed(1234)

model = a2c()

class a2c_agent():
    def __init__(self):
        self.env=gym.make(env_name)
        self.viewer = rendering.SimpleImageViewer()
        self.render=render
        self.episode=1
        self.total_reward=0
        self.clear()
        self.EPS=eps_start
        if create_video:    
            self.env = wrappers.Monitor(self.env, dir_,force=True)
        print('observations shape:',observation_shape)
        #print('action details', action_details)
        print('no of actions:', no_of_actions)
    def getName(self):
            return self.__class__.__name__
    def run_episode(self):
        observation = self.env.reset()
        t = 0   #to calculate time steps
        while True:
            if (self.render):
                rgb=self.env.render('rgb_array')#show the game, xlib error with multiple threads
                #upscale and show the game
                upscaled= rescale(rgb,4,4)
                self.viewer.imshow(upscaled)
                #visualization of lenet layers
                out=model.visualize([observation])
                plotter(out)
                if (mode=='test'):
                    time.sleep(0.2)
            if use_model =='human':
                self.record()
                action=self.action
                #print("step:",t,end="\r")
            else:
                action_prob=model.predict_action_prob([observation])
                action=self.predict_action(action_prob,t)
            self.observations.append(observation)
            self.actions.append(onehot(action,no_of_actions))
            observation_new, reward, done, info = self.env.step(action)
            self.total_reward+=reward   
            self.r.append(reward)   
            t=t+1
            observation=observation_new
            if done:
                self.R_terminal=0
                self.bellman_update() #can be used for batch
                print(" episode:",self.episode,"eps:","{0:.2f}".format(self.EPS), " reward:",self.total_reward," took {} steps".format(t))
                self.R=np.reshape(self.R,(np.shape(self.R)[0],1))
                model.log_details(self.total_reward,self.observations,self.actions,self.R,self.episode,self.getName())
                self.train()
                self.total_reward=0
                break
            elif (small_batch==True):
                if (t%10 == 0):
                    self.R_terminal=model.predict_value([observation_new])
                    self.bellman_update()
                    self.train()

    def run(self):
        start = time.time()
        model.load()
        while self.episode<max_no_episodes:
            self.run_episode()
            # self.train()#for a batch of complete episode
            self.EPS-=d_eps
            self.episode+=1
            if self.episode%ckpt_episode==0:
                model.save(self.episode)
                print("saved model at episode {}".format(self.episode))
        end = time.time()
        print("took:",end - start)

    def train(self):
        self.R=np.reshape(self.R,[np.shape(self.R)[0],1])
        model.train_actor(self.observations,self.actions,self.R)
        model.train_critic(self.observations,self.R)
        self.clear()

    def clear(self):
        self.observations=[]
        self.actions=[]
        self.R=[]
        self.r=[]
        self.R_terminal=0

    def bellman_update(self):
        self.R=[]
        for i in range(len(self.r),0,-1):
            t=self.r[i-1]+self.R_terminal
            self.R.append([t])
            #self.R_terminal=t #normal bellman update
            self.R_terminal=model.predict_value([self.observations[i-1]]) #batch bootstrap
        self.R=np.flip(self.R,axis=0)   
    
    def predict_action(self,prob,t):
        #here we use epsilon greedy exploration by tossing a coin
        action=np.argmax(prob)
        if np.random.uniform() < self.EPS:
            action=self.env.action_space.sample()
        return action
    def test(self):
        model.load()
        self.render=True
        self.run_episode()
        print("done..")


