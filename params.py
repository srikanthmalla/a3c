#openai gym environments
environment="Breakout-v4"
create_video= False
# env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_ACTIONS = 4

#tensorflow details
batch_size=1
tf_logdir='./graphs/results'
learning_rate=1E-3

#coefficients
LOSS_V=0.5
LOSS_ENTROPY=0.01

#atari_breakoutV0
input_shape=(None,210, 160, 3)
output_shape=4

#asynchronous 
THREADS=2
OPTIMIZERS=2
RUN_TIME=30
max_no_episodes=5

GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

#epsilon greedy, not the learning rate
EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000