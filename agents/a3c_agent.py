from agents.a2c_agent import *
class a3c_agent(a2c_agent,Thread):
    def __init__(self):
        a2c_agent.__init__(self)
        Thread.__init__(self)
    def getName(self):
        return Thread.getName(self)
