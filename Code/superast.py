import numpy as np
import time
import os
import multiprocessing as mp

#Constants##########################################################################
G = 6.67408 * 1e-11
Re = 6371 * 1e3
M = 5.972 * 1e24
m = 419455 # 419455 for ISS
H = np.array([300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]) * 1e3
dt = 1

Area = 125600
Cd = 0.47
Ro = np.array([2.58e-11, 1.72E-11, 1.16e-11, 7.99E-12, 5.55e-12, 3.89E-12, 2.75e-12, 1.96E-12, 1.40e-12, 1.01E-12, 7.30e-13])

#MAIN LOOP##########################################################################

def execute(h0, ro, drop=10e3):
    R = Re + h0
    p = np.array([R, 0])
    v = 1 * np.array([0, np.sqrt(G*M/R)]) #||v|| = 7.66km/s for ISS 
    a = 0
    t = 0

    h = h0
    while(h > (h0 - drop) ):
        '''
        print('Acceleration: ', a)
        print('Velocity: ', v)
        print("Position: ", p)
        print('Height:', h)
        print('ro:', ro)
        print('t:', t)
        print()
        '''

        R = np.linalg.norm(p)
        a = - p * G*M/(R**3) - np.linalg.norm(v)*v*ro*Area*Cd/(2*m)
        v += a * dt
        p += v * dt
        t += dt

        h = np.linalg.norm(p)-Re
    
    return t/(60*60*24)


start_time = time.time()

pool = mp.Pool(mp.cpu_count())
results = pool.starmap(execute, [(H[i], Ro[i]) for i in range(9,10)])
pool.close()

print("Execution time with parallelization:", time.time() - start_time)



#print(t/(60*60*24))
