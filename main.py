from params import *
from models import Environment,Optimizer
import time

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for env in envs:
	env.start()
# for opt in opts:
# 	opt.start()

# time.sleep(RUN_TIME)

# for env in envs:
# 	env.stop()
# for env in envs:
# 	env.join()

# for opt in opts:
# 	opt.stop()
# for opt in opts:
# 	opt.join()
# print("Training finished")
