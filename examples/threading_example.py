from threading import Thread

#threading on single class with only one inheritance
class cat(Thread):
    def __init__(self):
        super(cat,self).__init__()
        self.weight=10
    def shout(self):
        print("%s meow"%(self.getName()))
    #this method will be used by instance.start()
    def run(self):
        for i in range(5):
            self.shout()
#threading with multiple inheritance
#to use super we need to use object for which we should not use args for init
#if you don't want just seperately initialise all the parent classes
class dog():
    def __init__(self):
        #super(dog,self).__init__()
        self.weight=30
    def shout(self):
        print("bow")
class hound(dog,Thread):
    def __init__(self):
        #super(hound,self).__init__()
        dog.__init__(self)
        Thread.__init__(self)
        self.weight=60 #overwriting the dog weight
    def shout(self):
        print("%s bow"%(self.getName()))
    def run(self):
        for i in range(4):
            self.shout()

d=dog()
d.shout()

if __name__=='__main__':
    cats=[hound() for i in range(10)] #comprehension
    for cat in cats:
        cat.start()
    #join method is to wait for the threads to finish
    for cat in cats:
        cat.join()
    print('done..')
