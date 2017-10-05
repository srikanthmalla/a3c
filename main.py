from params import *
from agents import *

if use_model=='a2c':
    Agent = a2c_agent()
elif use_model=='a3c':
    agents=[a3c_agent() for i in range(THREADS)]
    for agent in agents:
        agent.start()
    for agent in agents:
        agent.join()
else:
    Agent = trpo_agent()
	
Agent.run()
print("Training Finished...")



