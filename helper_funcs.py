import numpy as np

def print_shape(x):
    print(x.name,':',x.get_shape())
def onehot(action,no_of_actions):
    x=np.zeros(no_of_actions)
    x[action]=1
    return x
def rescale(rgb,y,x):
	return np.repeat(np.repeat(rgb, y, axis=0), x, axis=1)
def match(matrix,shape):
	M=np.zeros(shape)
	M[0:np.shape(matrix)[0]]=matrix
	return M

import matplotlib.pyplot as plt
# import matplotlib.animation as animation
fig = plt.figure()
def plotter(img):
	plt.ion()
	# ax = fig.add_subplot(111)
	# li, = ax.plot(index_list, reward_list,'-')	
	# draw and show it
	# fig.canvas.draw()
	plt.imshow(img,cmap='gray')
	# plt.show(block=False)
	plt.pause(0.05)
