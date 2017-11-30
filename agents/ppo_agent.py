from agents.a2c_agent import a2c_agent
from models.ppo import ppo

model=ppo()
class ppo_agent(a2c_agent):
    def __init__(self):
        a2c_agent.__init__(self)
