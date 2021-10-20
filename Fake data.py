#!/usr/bin/env python
# coding: utf-8

# In[36]:


import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import animation
from IPython.display import HTML

x, y, z= 100, 100, 200 #z is stack 
stack_data = np.zeros((z, y, x), 'uint8')
video=np.random.normal(10, 1, [z, y, x])
#plt.imshow(stack_data_noise[3])
plt.axis('off')
def init():
    im.set_data(video[0,:,:])
def animate(i):
    im.set_data(video[i,:,:])
    return im
def disp_video_ipython(video, playback_frame_rate = 10):
    fig = plt.figure()
    im = plt.imshow(video[0,:,:])
    plt.axis('off')
    plt.close() # this is required to not display the generated image
    interval = 100 * 1/playback_frame_rate
    print(interval)
    return im, animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                interval=interval, repeat=True) #interval is the interval between frame
im, anim=disp_video_ipython(video)
HTML(anim.to_html5_video())


# In[ ]:




