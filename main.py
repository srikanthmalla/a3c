from params import *

if use_model=='dqn':
    from agents.dqn_agent import *
    Agent = dqn_agent()
elif use_model=='a2c':
    from agents.a2c_agent import *
    Agent = a2c_agent()
elif use_model=='ppo':
    from agents.ppo_agent import *
    Agent=ppo_agent()

if mode=='test':
    Agent.test()
else:
    if use_model=='human':
        from agents.human_agent import human_agent
        Agent = human_agent()
        Agent.run()
    elif use_model=='a3c':
        agents=[a3c_agent() for i in range(THREADS)]
        for agent in agents:
            agent.start()
        for agent in agents:
            agent.join()
    else:
        Agent.run()
print("Training Finished...")



