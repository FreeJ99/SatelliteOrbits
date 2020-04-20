import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Constants
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
h = 3000 * 1e3
dt = 60

viewSize = 3*Re

#Initial values
R = Re + h
p0 = np.array([R, 0])
v0 = np.array([0, np.sqrt(G*M/R)*1])

print(v0)

#Animation
fig = plt.figure() 
ax = plt.axes(xlim=(-viewSize, viewSize), ylim=(-viewSize, viewSize)) 
line, = ax.plot([], [], marker='o', color='orange') 

def init():
    circle = plt.Circle((0,0), radius=Re, color='lightblue')
    ax.add_artist(circle)
    line.set_data([], []) 
    return line, 

p = p0
v = v0
a = None

def printValues():
    print('Acceleration: ', a)
    print('Velocity: ', v)
    print('Position:', p)
    print()

def updateValues():
    global v
    global p
    global a

    R = np.linalg.norm(p)
    nv = np.array([v[1], -v[0]]) / np.linalg.norm([v[1], -v[0]])
    ps = p/np.linalg.norm(p)
    a = - ps * G*M/(R**2) #nv * (v**2)/R #problem with numerical precision???
    #a = np.zeros(2)
    v += a * dt
    p += v * dt

    return p

def animate(i):
    if(i%10 == 0):
        printValues()

    p = updateValues()

    line.set_data([p[0]], [p[1]])
    return line, 

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                            frames=365, interval=40, blit=True)

plt.show()