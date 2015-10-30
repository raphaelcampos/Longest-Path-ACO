import AntColonyOptimization as aco

weight_matrix, b_node, e_node = aco.load_graph_file('entradas/entrada2.txt')

colony = aco.AntColony(200, 5000, 0.5)

best_ant = colony.meta_heuristic(weight_matrix, b_node, e_node)
print best_ant.path_, best_ant.path_length_

#ant = aco.Ant()

#colony.init_pheromone_matrix_(weight_matrix)
#ant.random_walk(colony.pheromone_matrix_, weight_matrix, b_node, e_node)