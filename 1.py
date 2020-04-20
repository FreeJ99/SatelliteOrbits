import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

rcParams['figure.figsize'] = [12, 12]

#Constants
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
m = 10000
h = 3000 * 1e3
dt = 60

viewSize = 3*Re

#Initial values
R = Re + h
p0 = np.array([R, 0])
v0 = np.array([0, np.sqrt(G*M/R)*1.1])
t = 0

print(v0)

#Figures, axes and lines
fig, ((axSatellite, axPotential), (axKinetic, _)) = plt.subplots(2, 2)
axSatellite.set_xlim(-viewSize, viewSize)
axSatellite.set_ylim(-viewSize, viewSize)

axPotential.set_xlabel('time')
axPotential.set_ylabel('Energy')

lSatellite, = axSatellite.plot([], [], marker='o', color='orange') 
lPotential, = axPotential.plot([], [], color='lightblue', label='Potential')
lKinetic, = axKinetic.plot([], [], color='orange', label='Kinetic')
axPotential.set_xlim(0, 1e4)
axPotential.legend()
axKinetic.set_xlim(0, 1e4)
axKinetic.legend()

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
    global t

    R = np.linalg.norm(p)
    #nv = np.array([v[1], -v[0]]) / np.linalg.norm([v[1], -v[0]])
    a = - p * G*M/(R**3) #nv * (v**2)/R #problem with numerical precision???
    v += a * dt
    p += v * dt
    t += dt

    Ep = -G * m * M / R 
    Ek = m*(np.linalg.norm(v)**2) / 2

    return (p, Ep, Ek)

def init():
    #print('init')
    circle = plt.Circle((0,0), radius=Re, color='lightblue')
    axSatellite.add_artist(circle)
    #lSatellite.set_data([], []) 
    #print('endinaxPotential.set_xlim(0, 1e3)it')
    return lPotential, lSatellite, lKinetic

def rescaleAx(ax, x, y):
    #TODO: change ticks
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)

    oldX = ax.get_xlim()
    oldY = ax.get_ylim()
    #print(oldY)
    #print(min_y, max_y)
    if(max_x > oldX[1]):
        ax.set_xlim(oldX[0], 5*oldX[1])
    if((abs(min_y) < abs(oldY[0])) or (abs(max_y > abs(oldY[1])))):
        #print('scaling y')
        ax.set_ylim(oldY[0]-abs(oldY[0]), oldY[1]+abs(oldY[1]))

def animate(i):
    #print('animate')
    if(i%10 == 0):
        printValues()
        axPotential.set_ylim(-1e11, -1e12)
    
    p, Ep, Ek = updateValues()
    lSatellite.set_data([p[0]], [p[1]])
    
    #print('lPotential', type(lPotential))

    x = lPotential.get_xdata()
    x = np.append(x, t)
    y = lPotential.get_ydata()
    y = np.append(y, Ep)
    rescaleAx(axPotential, x, y)
    lPotential.set_data(x, y)

    x = lKinetic.get_xdata()
    x = np.append(x, t)
    y = lKinetic.get_ydata()
    y = np.append(y, Ek)
    rescaleAx(axKinetic, x, y)
    lKinetic.set_data(x, y)

    return lPotential, lSatellite, lKinetic

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                            frames=365, interval=40, blit=True)

plt.show()