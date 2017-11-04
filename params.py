use_model='a2c'#human, a2c, a3c
mode='train'
#openai gym environments
#--------
env_name="CartPole-v0"
use_net='fc_1'#fc_1, lenet, VGG
#---------
#env_name="Breakout-v0"
#use_net='lenet'
#-------
import gym
Environment=gym.make(env_name)
no_of_actions= Environment.action_space.n
observation_shape=Environment.observation_space.shape
render=False
##--------------
#action_details=Environment.unwrapped.get_action_meanings() #cartpole doesnt has this action meanings but breakout does
#print(action_details)
#print(Environment.action_space.sample())
##--------------
#tensorflow details
input_shape=(None,)+observation_shape
output_shape = (None,)+ (no_of_actions,)
batch_size=1
actor_lr=1E-3
critic_lr=1E-3
tf_logdir='./graphs/aclr_'+str(actor_lr)+',cr_lr'+str(critic_lr)+'/'
LOSS_V=100.0#should be float
dir_="./tmp/"+env_name+"/"+use_model
import os
if not os.path.exists(dir_):
    os.makedirs(dir_)
ckpt_dir=dir_+"/model.ckpt"
#RL_agent details
if use_model=='human':
    max_no_episodes=3
    ckpt_episode=1
else:
    max_no_episodes=2000
    ckpt_episode=100

if env_name=='Breakout-v0':
    GAMMA=1
else:
    GAMMA = 0.99

#epsilon greedy, not the learning rate
if mode=='train':
    eps_start = 0.9
    create_video= False
else:
    eps_start = 0.0
    create_video= True

eps_stop  = 0.0
eps_steps = max_no_episodes
d_eps= (eps_start-eps_stop)/eps_steps

#A3C details
THREADS=2

##networks
if use_net=='fc_1':
    from networks.FCN_one_hidden import *
    net=FCN_one_hidden
elif use_net=='lenet':
    from networks.lenet import *
    net=lenet
elif use_net=='VGG':
    from networks.VGG import *
    net=VGG

##observations
# Episodic batch with lr=E-2, Loss_v=100, Epochs=2K, A2C algo works on Cart-pole. 
# To work with A2C on cart-pole, when learning online with batch of 10 it takes 3K epochs with Loss_V=1000 and lr=E-3
# To work with A3C on cart-pole, When learning online with batch of 10 it takes 2K epochs with Loss_V=1000 and lr=E-2, THREADS=2  
