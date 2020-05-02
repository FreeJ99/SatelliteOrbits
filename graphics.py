#Note: plotted earth has 0.8 of it's radius

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import time

rcParams['figure.figsize'] = [12, 12]

#Constants##########################################################################
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
m = 4194 # 419455 for ISS
h = 408 * 1e3 #408*1e3 for ISS
dt = 60

viewSize = 2*Re

Area = 7850
Cd = 0.47

#Initial values######################################################################
R = Re + h
p = np.array([R, 0])
v = 1 * np.array([0, np.sqrt(G*M/R)])
t = 0
a = None
ro = 6.28e-11
roGnd = 1.23

#Figures, axes and lines##############################################################
fig, ((axSatellite, axEnergy), (axHeight, axVelocity)) = plt.subplots(2, 2)

axSatellite.set_xlim(-viewSize, viewSize)
axSatellite.set_ylim(-viewSize, viewSize)

axEnergy.set_title("Energy")
axEnergy.set_xlabel('t')
axEnergy.set_ylabel('E')

axHeight.set_title("Height")
axHeight.set_xlabel('t')
axHeight.set_ylabel('h') 

axVelocity.set_title("Velocity")
axVelocity.set_xlabel('t')
axVelocity.set_ylabel('v')

circle = plt.Circle((0,0), radius=Re*0.7, color='lightblue', alpha=0.2)
axSatellite.add_artist(circle)

lSatellite, = axSatellite.plot([], [], marker='o', color='orange') 
lEnergy, = axEnergy.plot([], [], color='lightblue')
lHeight, = axHeight.plot([], [], color='orange')
lVelocity, = axVelocity.plot([], [], color='orange')

axEnergy.set_xlim(0, 1e4)
axHeight.set_xlim(0, 1e4)
axVelocity.set_xlim(0, 1e4)

#UPDATING VALUES###########################################################################
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
    global BC
    global ro

    h = np.linalg.norm(p)-Re
    #if(roGnd*np.exp(-h/1e4) > ro): #true when h<237km
     #   ro = roGnd*np.exp(-h/1e4)

    R = np.linalg.norm(p)
    a = - p * G*M/(R**3) - np.linalg.norm(v)*v*ro*Area*Cd/(2*m)
    v += a * dt
    p += v * dt
    t += dt

    Ep = -G * m * M / R 
    Ek = m*(np.linalg.norm(v)**2) / 2
    E = Ep + Ek

    return (p, E, h, np.linalg.norm(v), t)

#PLOTTING##########################################################################
def init():
    return lSatellite, lEnergy, lHeight, lVelocity

def rescaleAx(ax, x, y):
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)

    oldX = ax.get_xlim()
    oldY = ax.get_ylim()

    updated = False
    if(max_x > oldX[1]):
        ax.set_xlim(0, 5*oldX[1])
        updated = True 

    if((min_y < oldY[0]) or (max_y > oldY[1])):
        w = 1 + max_y - min_y#0.5e10
        ax.set_ylim(min_y-w, max_y+w)
        updated = True 

    
    if updated:
        global anim
        anim.event_source.stop()
        plt.draw()
        anim.event_source.start()

def appendToLine(line, ax, data):
    x = line.get_xdata()
    x = np.append(x, data[0])
    y = line.get_ydata()
    y = np.append(y, data[1])
    rescaleAx(ax, x, y)
    line.set_data(x, y)

def animate(i):
    #printValues()
    p, E, h, v, t = updateValues()

    lSatellite.set_data([p[0]], [p[1]])

    appendToLine(lEnergy, axEnergy, (t, E))
    appendToLine(lHeight, axHeight, (t, h))
    appendToLine(lVelocity, axVelocity, (t, v))
    
    return lSatellite, lEnergy, lHeight, lVelocity


plt.show()