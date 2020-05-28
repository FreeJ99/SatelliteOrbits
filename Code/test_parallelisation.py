import time
import multiprocessing as mp
import itertools

import numpy as np

from pendulum.physics import ODEs, Model

from pendulum.compute import compute, save

def get_time(time_model):
	if time_model:
		return time.time()
	else:
		return time.perf_counter()

test_number = 1

runOnlyParallel = False
cpus = min(mp.cpu_count(), 10)
points = 10
all_shifts = False

n = 3
dt = 1.0/60.0
t_upper = 90.0
t = np.arange(0.0, t_upper, dt)

shift_positions = n
max_angle_shift = np.pi * 1e-3
shift_angles = np.linspace(-max_angle_shift, max_angle_shift, 10)
angle_positions = len(shift_angles)

odes = ODEs(Model(n))

masses = np.array([1.0, 1.0, 1.0])
lengths = np.array([2.0, 2.0, 2.0])
initial_angles = np.deg2rad(np.array([-140.0, -140.0, -140.0]))
initial_velocities = np.deg2rad(np.array([0.0, 0.0, 0.0]))

def compute_result_single(shift, position):
	shifted_angles = initial_angles
	shifted_angles[position] += shift
	return compute(masses, lengths, shifted_angles, initial_velocities, t, odes)

def compute_result_all(shifts):
	shifted_angles = initial_angles
	for lhs, rhs in zip(shifted_angles, shifts):
		lhs += rhs
	return compute(masses, lengths, shifted_angles, initial_velocities, t, odes)

outpath_save_result = None

def wrapper_single(tpl):
	shift, position = tpl
	res = compute_result_single(shift_angles[shift], position)
	save(res, outpath_save_result.format(test_number, '{}_{}'.format(shift, position)))
	return res

def wrapper_all(tpl):
	shift_1, shift_2, shift_3 = tpl
	shifts = map(lambda i: shift_angles[i], tpl)
	res = compute_result_all(shifts)
	save(res, outpath_save_result.format(test_number, '{}_{}_{}'.format(shift_1, shift_2, shift_3)))
	return res

time_model = True
time_total_start = get_time(time_model)
time_processings = []

def compute_parallel():
	res = None
	pool = mp.Pool(cpus)
	if all_shifts:
		res = pool.map_async(wrapper_all, itertools.product(range(angle_positions), range(angle_positions), range(angle_positions)), chunksize=cpus)
	else:
		res = pool.map_async(wrapper_single, itertools.product(range(angle_positions), range(shift_positions)), chunksize=cpus)
	pool.close()
	pool.join()
	return res

def compute_iterative():
	res = []
	if all_shifts:
		for tpl in itertools.product(range(angle_positions), range(angle_positions), range(angle_positions)):
			tmp = wrapper_all(tpl)
			res.append(tmp)
	else:
		for tpl in itertools.product(range(angle_positions), range(shift_positions)):
			tmp = wrapper_single(tpl)
			res.append(tmp)
	return res

outpath_stats = 'data/stats/ode_solving/test_parallelisation_{}.txt'
different_pendulum_count = angle_positions ** 3 if all_shifts else shift_positions * angle_positions

f = open(outpath_stats.format(test_number), 'w')

outpath_save_result = 'data/results/test_parallelisation_{}_parallel_{}.out'

print('Beginning parallel calculations')
time_ode_solve_parallel_start = get_time(time_model)
res = compute_parallel()
time_ode_solve_parallel_end = get_time(time_model)
delta_time = time_ode_solve_parallel_end - time_ode_solve_parallel_start
f.write('Parallel execution with {} processes for {} pendulums took {} seconds\n'.format(cpus, different_pendulum_count, delta_time))

if not runOnlyParallel:
	print('Beginning iterative calculations')
	outpath_save_result = 'data/results/test_parallelisation_{}_iterative_{}.out'
	time_ode_solve_iterative_start = get_time(time_model)
	res = compute_iterative()
	time_ode_solve_iterative_end = get_time(time_model)
	delta_time = time_ode_solve_iterative_end - time_ode_solve_iterative_start
	f.write('Iterative execution with {} processes for {} pendulums took {} seconds\n'.format(cpus, different_pendulum_count, delta_time))


time_total_end = get_time(time_model)

print('Finished ODEs')

f.write('Total execution took {} seconds.\n'.format(time_total_end - time_total_start))
f.close()

# outpath_vid = 'data/video/test_parallelisation_{}.mkv'
# generate_animation(res, multi=True, show=False, outpath=outpath_vid.format(test_number))

# print('Generated video')
