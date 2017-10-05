#openai gym environments
env_name="CartPole-v0"
import gym
Environment=gym.make(env_name)
create_video= False
no_of_actions= Environment.action_space.n
observation_shape=Environment.observation_space.shape
#action_details=Environment.unwrapped.get_action_meanings() #cartpole doesnt has this action meanings but breakout does
render=False

#tensorflow details
input_shape=(None,)+observation_shape
output_shape = (None,)+ (no_of_actions,)
batch_size=1
actor_lr=1E-1
critic_lr=1E-1
tf_logdir='./graphs/aclr_'+str(actor_lr)+',cr_lr'+str(critic_lr)+'/'
LOSS_V=1.0

#RL_agent details
use_model='a3c'
max_no_episodes=1000
ckpt_episode=100
GAMMA = 0.99

#epsilon greedy, not the learning rate
eps_start = 0.9
eps_stop  = 0.0
eps_steps = max_no_episodes
d_eps= (eps_start-eps_stop)/eps_steps

#A3C details
THREADS=2
BATCH_SIZE=100
