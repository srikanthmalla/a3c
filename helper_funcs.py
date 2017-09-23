import numpy as np

def print_shape(x):
    print(x.name,':',x.get_shape())
def onehot(action,no_of_actions):
    x=np.zeros(no_of_actions)
    x[action]=1
    return x
