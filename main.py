from params import *
from agents import *

if use_model=='a2c':
	Agent = a2c_agent()
elif use_model=='a3c':
	Agent = a3c_agent()
else:
	Agent = trpo_agent()
	
Agent.run()
print("Training Finished...")



