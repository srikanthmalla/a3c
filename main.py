from params import *
from models import Environment

envs = [Environment() for i in range(threads)]

for env in envs:
	env.start()
