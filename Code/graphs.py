import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import time
import os

rcParams['figure.figsize'] = [12, 9]

program_start_time = time.time()

#Constants##########################################################################
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
m = 419455 # 419455 for ISS
h = 4.08 * 1e5#408*1e3 for ISS
dt = 1

Area = 125600
Cd = 0.47

#Initial values######################################################################
R = Re + h
p = np.array([R, 0])
v = 1 * np.array([0, np.sqrt(G*M/R)]) #||v|| = 7.66km/s for ISS 
t = 0
a = 0
ro = 3.3e-12 #http://www.braeunig.us/space/atmos.htm


#Figures and axes##############################################################
figEnergy, axEnergy = plt.subplots()
figHeight, axHeight = plt.subplots()
figVelocity, axVelocity = plt.subplots()

axEnergy.set_title("Energy")
axEnergy.set_xlabel('t (min)')
axEnergy.set_ylabel('E')

axHeight.set_title("Height")
axHeight.set_xlabel('t (min)')
axHeight.set_ylabel('h (km)') 

axVelocity.set_title("Velocity")
axVelocity.set_xlabel('t (days)')
axVelocity.set_ylabel('v (m/s)')

Es = []
Hs = []
Vs = []
ts = []

def savePlots(base, figures, paths):
    for fig, p in zip(figures, paths):
        fig.savefig(os.path.join(base,p))
#DEBUGGING###########################################################################
def printValues():
    print('Acceleration: ', np.linalg.norm(a))
    print('Velocity: ', np.linalg.norm(v))
    print('Height:', h)
    print('ro:', ro)
    print('t:', t)
    print()

#MAIN LOOP##########################################################################
h = np.linalg.norm(p) - Re
while(h > 4.0e5):
    if(t%100000 == 0 or h<25130):
        printValues()

    R = np.linalg.norm(p)
    a = - p * G*M/(R**3) - np.linalg.norm(v)*v*ro*Area*Cd/(2*m)
    v += a * dt
    p += v * dt
    t += dt

    Ep = -G * m * M / R 
    Ek = m*(np.linalg.norm(v)**2) / 2
    E = Ep + Ek

    h = np.linalg.norm(p)-Re

    Es.append(E)
    Hs.append(h/1e3)
    Vs.append(np.linalg.norm(v))
    ts.append(t/(60*60*24))

print("Execution time:", time.time() - program_start_time)

axEnergy.plot(ts, Es, color='lightblue')
axHeight.plot(ts, Hs, color='lightgreen')
axVelocity.plot(ts, Vs, color='orange')

savePlots(base = '../Small_dt', figures=[figEnergy, figHeight, figVelocity], 
            paths=['E', 'H', 'V'])

plt.show()