import time
import keyboard
class key():
    def record(self):
        keyboard.start_recording()
        time.sleep(4)
        recorded=keyboard.stop_recording()
        if recorded[0].name=='right':
            print('right')
if __name__=='__main__':
    k=key()
    k.record()
