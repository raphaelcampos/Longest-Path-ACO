import numpy as np
import math
from operator import attrgetter
import copy

import heapq

import matplotlib.pyplot as plt

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


class AntColony(object):

	"""docstring for AntColony"""
	def __init__(self, n_ants = 100, iterations = 100, evaporation_rate = 0.3, k = 10):
		super(AntColony, self).__init__()
		self.pheromone_matrix_ = None
		
		self.iterations = iterations
		self.evaporation_rate = evaporation_rate

		self.n_ants = n_ants
		self.k = k


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
			
			#print i, best_ant.path_length_
			
			# reinforcing the best path of the current iteration
			#idx, delta = best_ant.release_pheromone()
			#self.pheromone_matrix_[idx] += self.n_ants*delta
			#print delta
			
			last[last_idx] = best_ant.path_length_
			last_idx = (last_idx + 1)%len(last)
			# early stop
			if np.std(last) == 0:
				break


		return max(best_ants, key=attrgetter('path_length_'))

	def generate_solutions_(self, weight_matrix, b_node, e_node):
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

	def init_pheromone_matrix_(self, weight_matrix, b_node, e_node, epsilon = 1.0):
		if not isinstance(weight_matrix, np.ndarray):
			raise Exception('weight_matrix must be of ndarray type')
		
		self.pheromone_matrix_ = (weight_matrix != 0) * float(epsilon)

		self.generate_solutions_(weight_matrix, b_node, e_node)

		aux = self.evaporation_rate
		self.evaporation_rate = 0.0;
		self.pheromone_update_(self.k)
		self.evaporation_rate = aux



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
		p = p[nin_path]/float(p[nin_path].sum())

		return np.random.choice(np.arange(n)[nin_path], 1, p=p)[0]

	def __cmp__(self, other):
		return cmp(self.path_length_, other.path_length_)