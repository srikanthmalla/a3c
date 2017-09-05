from params import *
from models import Environment,Optimizer
import time

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

#start multiple agents on multiple threads
for env in envs:
	env.start()

#start multiple optimizers on multiple threads
for opt in opts:
	opt.start()

#if the time is too much
time.sleep(RUN_TIME)

#send the stop signal
for env in envs:
	env.stop()

#wait until every thread is done
for env in envs:
	env.join()

#stop the optimizers
for opt in opts:
	opt.stop()

# wait for all the threads to stop
for opt in opts:
	opt.join()
print("Training finished")
