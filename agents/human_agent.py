from agents.a2c_agent import *
import keyboard
#this human agent mapping is only for breakout
class human_agent(a2c_agent):
    def __init__(self):
        a2c_agent.__init__(self)
        self.render=True
        self.action=0
    def record(self):
        keyboard.start_recording()
        time.sleep(0.4)
        recorded=keyboard.stop_recording()
        try:
            action=recorded[0].name
        except IndexError:
            action=None

        if action=='right':
            self.action=2
        elif action=='left':
            self.action=3
        elif action=='up':
            self.action=1
        else: 
            self.action=0
    def run(self):
        start = time.time()
        while self.episode<max_no_episodes:
            self.run_episode()
            self.episode+=1
            if self.episode%ckpt_episode==0:
                model.save(self.episode)
                print("saved model at episode {}".format(self.episode))
        end = time.time()
        print("took:",end - start)
