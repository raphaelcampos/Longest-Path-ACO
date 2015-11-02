#!/usr/bin/env python
import copy
import heapq
import math

import numpy as np
import matplotlib.pyplot as plt

from operator import attrgetter
from prettytable import PrettyTable
 
from threading import Thread

from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import Process, Queue

import time

default_nprocs = cpu_count()

def load_graph_file(file_str):
	with open(file_str, 'r') as f:
		data = f.readlines()
		n = int(data[0])
		weight_matrix = np.zeros((n, n))
		obj = np.array(data[1].split(), int) - 1

		for line in data[2:]:
			tupla = line.split()
			edge = (int(tupla[0])-1, int(tupla[1])-1)
			weight_matrix[edge[0], edge[1]] = float(tupla[2])

	return weight_matrix, obj[0], obj[1]

def distribute(nitems, nprocs=None):
    if nprocs is None:
        nprocs = default_nprocs
    nitems_per_proc = (nitems+nprocs-1)/nprocs
    return [(i, min(nitems, i+nitems_per_proc))
            for i in range(0, nitems, nitems_per_proc)]

def do_random_walk(ants, pheromone_matrix, weight_matrix, b_node, e_node, q):
	for ant in ants:
		ant.random_walk(pheromone_matrix, weight_matrix, b_node, e_node)
		q.put(ant)

class Extractor(object):
	"""docstring for Extractor"""
	def __init__(self, header=False, comments="#", delimiter=" "):
		super(Extractor, self).__init__()
		self.header = header
		self.comments = comments
		self.delimiter = delimiter

	def extract(self, file_str):
		ext = []
		with open(file_str, 'r') as f:
			data = f.readlines()
			if self.header:
				ext.append(data[0].split(self.comments)[1])
				data = data[1:]

			dump = None
			for d in data:
				d = np.array(d.split(self.delimiter), dtype=float)
				dump = np.vstack((dump, d)) if not (dump == None) else d

			ext.append(dump)
			
		return ext

class Outputer(object):
	"""docstring for Outputer"""
	def __init__(self):
		super(Outputer, self).__init__()

	def format(self, input_obj):
		if isinstance(input_obj, Logger):
			return self._format_logger(input_obj)
		elif isinstance(input_obj, str):
			return self._format_output(input_obj)
		else:
			raise Exception('Method cannot handle input.')
	
	def _format(self, setup, dump):
		out = ""

		params = PrettyTable(["Parameter", "Value"])
		for k, v in setup.iteritems():
			params.add_row([k,v])

		out += params.get_string()
		out = out + "\n\n"

		split_points = np.where(dump[:,0] == 0)[0][1:]
		
		if split_points == []:
			trails = [dump]		
		else:
			trails = np.split(dump, split_points)

		results = PrettyTable(["Trial", "Iterations", "Best solution"])		
		n_trials = len(trails)
		stops = np.zeros(n_trials)
		best_solution = np.zeros(n_trials)
		for i, t in enumerate(trails):
			stops[i] = t[-1,0]
			best_solution[i] = max(t[:,1])
			results.add_row([i + 1, stops[i], best_solution[i]])

		results.add_row(["Avg", np.average(stops), np.average(best_solution)])
		
		results
		out += results.get_string()  

		return out

	def _format_logger(self, logger):
		return self._format(logger.setup, logger.dump)

	def _format_output(self, output_file, extractor=Extractor(header=True)):
		setup, dump = extractor.extract(output_file)
		return self._format(eval(setup), dump)

class Logger(object):
	"""docstring for Logger"""
	def __init__(self, dump_file="", append=False, np_save=True, outputer=None):
		super(Logger, self).__init__()
		
		self.dump_file = dump_file
		self.append = append
		self.np_save = np_save
		self.dump = None
		self.f = None

		if outputer == None:
			self.outputer = Outputer()
		else:
			self.outputer = outputerself.outputer

	def init(self):
		self.reset()
		if self.append and not (self.dump_file == ""):
			self.f=open(self.dump_file, 'ab')

		self.first = True


	def reset(self):
		self.dump = None
		if self.f:
			self.f.close()

	def running_setup(self, setup):
		self.setup = setup
		
	def collect(self, dump_iteration):
		if self.append:
			if self.f.closed:
				self.f=open(self.dump_file, 'ab')

			if self.first:
				header = str(self.setup)
				np.savetxt(self.f, dump_iteration, fmt='%d',header=header)
				self.first = False
			else:
				np.savetxt(self.f, dump_iteration, fmt='%d')
			
		if not (self.dump == None):
			self.dump = np.vstack((self.dump, dump_iteration))
		else:
			self.dump = dump_iteration

	def running_summary(self):
		return self.outputer.format(self)

class AntColony(object):

	"""docstring for AntColony"""
	def __init__(self, n_ants = 100, iterations = 100, evaporation_rate = 0.3, k = 10, n_jobs = 1, logger=None, random_state=None):
		super(AntColony, self).__init__()
		self.pheromone_matrix_ = None
		
		self.iterations = iterations
		self.evaporation_rate = evaporation_rate

		self.n_ants = n_ants
		self.k = k

		self.n_jobs = default_nprocs if n_jobs < 1 else n_jobs

		# initialize logger if one is given
		if not (logger == None):
			logger.init()
		
		self.logger = logger

		np.random.seed(random_state)
		self.random_state = random_state

	def meta_heuristic(self, weight_matrix, b_node, e_node):
		self.create_ants_(self.n_ants)
		self.init_pheromone_matrix_(weight_matrix, b_node, e_node)
		
		last = np.zeros(20)
		last_idx = 0

		best_ants = []
		for i in xrange(self.iterations):
			self.generate_solutions_(weight_matrix, b_node, e_node)
			self.pheromone_update_(self.k)
			
			best_ant = max(self.ants_, key=attrgetter('path_length_'))
			best_ants.append(copy.copy(best_ant))
			
			# dump to log file if logger was initialized
			self._dump(i)

			last[last_idx] = best_ant.path_length_
			last_idx = (last_idx + 1)%len(last)
			# early stop
			if np.std(last) == 0:
				break

		return max(best_ants, key=attrgetter('path_length_'))

	def generate_solutions_(self, weight_matrix, b_node, e_node):
		if self.n_jobs > 1:
			q = Queue() 
			
			slices = distribute(self.n_ants, self.n_jobs)

			jobs = [Process(target=do_random_walk, args=(self.ants_[s:e], self.pheromone_matrix_, weight_matrix, b_node, e_node, q)) for (s,e) in slices]

			# Run processes
			for p in jobs:
				p.start()

			ants = []
			liveprocs = list(jobs)
			while liveprocs:
				try:
					while 1:
					    ants = ants + [(q.get(False))]
				except Exception, e:
					pass

				time.sleep(0.005)    # Give tasks a chance to put more data in
				if not q.empty():
					continue
				liveprocs = [p for p in liveprocs if p.is_alive()]

			self.ants_ = ants
		else:
			for ant in self.ants_:
				ant.random_walk(self.pheromone_matrix_, weight_matrix, b_node, e_node)
		
	def pheromone_update_(self, k_top = 0):
		p = self.pheromone_matrix_ * 0.0
	
		ants = self.ants_
		if k_top > 0:
			ants = heapq.nlargest( k_top, self.ants_ )

		for ant in ants:
			# it is not a good ant
			# therefore, it doesnt
			# deposit pheromone
			if ant.path_length_ == 1:
				continue

			idx, delta = ant.release_pheromone()			
			p[idx] += delta

		self.pheromone_matrix_ = (1 - self.evaporation_rate)*self.pheromone_matrix_ + p

	def create_ants_(self, n_ants):
		self.ants_ = []
		for i in range(n_ants):
			self.ants_.append(Ant())

	def init_ants(self, weight_matrix, b_node, e_node):
		for ant in self.ants_:
			ant.set_all(self.pheromone_matrix_, weight_matrix, b_node,e_node)

	def init_pheromone_matrix_(self, weight_matrix, b_node, e_node, epsilon = 1.0):
		if not isinstance(weight_matrix, np.ndarray):
			raise Exception('weight_matrix must be of ndarray type')
		
		self.pheromone_matrix_ = (weight_matrix != 0) * float(epsilon)

		self.generate_solutions_(weight_matrix, b_node, e_node)

		aux = self.evaporation_rate
		self.evaporation_rate = 0.0;
		self.pheromone_update_(self.k)
		self.evaporation_rate = aux

	def _dump(self, iteration):
		if not (self.logger == None):
			path_lengths = np.array([ant.path_length_ for ant in self.ants_])
			
			avg_path = np.average(path_lengths)
			std_path = np.std(path_lengths)
			best_ant = max(self.ants_, key=attrgetter('path_length_'))
			worst_ant = min(self.ants_, key=attrgetter('path_length_'))

			dump = [iteration, best_ant.path_length_,  worst_ant.path_length_, avg_path, std_path]
			self.logger.collect(np.array(dump).reshape(1,len(dump)))

class Ant(object):
	"""It's artificial agent responsible for walking
		on the graph from a node S to a node T.
		It performs a random walk accordingly to
		a transition matrix which is obtained from
		the pheromone matrix depending on the probability
		rule described in the method next_node_
	"""
	def __init__(self):
		super(Ant, self).__init__()
		self.path_length_ = 0
		self.path_ = None

	def release_pheromone(self):
		return self.path_.T.tolist(), ((1-1/float(self.path_length_)))

	def random_walk(self, pheromone_matrix_, weight_matrix, b_node, e_node):
		np.random.seed(None)
		nin_path = np.ones((pheromone_matrix_.shape[0]), dtype=bool)
		path_length = 0

		nin_path[b_node] = False
		next_node = self.next_node_(pheromone_matrix_, b_node, nin_path)
		path = np.array([[b_node, next_node]])
		path_length = path_length + weight_matrix[b_node, next_node]
		while not (next_node == e_node):
			current_node = next_node
			nin_path[current_node] = False

			try:
				next_node = self.next_node_(pheromone_matrix_, current_node, nin_path)
				path = np.append(path, [[current_node, next_node]], axis=0)
				path_length = path_length + weight_matrix[current_node,next_node]
			except Exception, e: 
				path_length = 1
				break

		self.path_length_ = path_length		
		self.path_= path
		

	def next_node_(self, pheromone_matrix_, c_node, nin_path):
		p = pheromone_matrix_[c_node, :]
		n = len(p)
		p_sum = float(p[nin_path].sum())

		if p_sum == 0.0:
			raise Exception()
		
		p = p[nin_path]/p_sum

		return np.random.choice(np.arange(n)[nin_path], 1, p=p)[0]

	def __cmp__(self, other):
		return cmp(self.path_length_, other.path_length_)