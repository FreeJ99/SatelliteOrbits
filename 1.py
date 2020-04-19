import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Constants
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
h = 3000 * 1e3

viewSize = 3*Re


#Initial values
R = Re + h
p0 = np.array([R, 0])
v0 = np.array([np.sqrt(G*M/R), 0])


#Animation
fig = plt.figure() 
ax = plt.axes(xlim=(-viewSize, viewSize), ylim=(-viewSize, viewSize)) 
line, = ax.plot([], [], marker='o', color='orange') 

def init():
    circle = plt.Circle((0,0), radius=Re, color='lightblue')
    ax.add_artist(circle)
    line.set_data([], []) 
    return line, 

frames = 365
def animate(i):
    t = (2*np.pi/frames) * i
    x = R * np.cos(t)
    y = R * np.sin(t)
    line.set_data([x], [y]) 
    return line, 

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                            frames=frames, interval=20, blit=True)

plt.show()