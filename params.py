#openai gym environments
environment="Breakout-v4"
create_video= False

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
threads=2
max_no_episodes=10