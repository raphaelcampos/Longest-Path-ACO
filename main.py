import AntColonyOptimization as aco

import argparse
parser = argparse.ArgumentParser(description="Heuristic to solve the longest path problem based on Ant Colony Optimization (ACO).")

parser.add_argument("input", type=str,
                    help="Input file")

parser.add_argument("-n", "--n_ants", type=int, help='Number of artificial ants per itereation (default: 200)', default=200)

parser.add_argument("-i", "--iterations", type=int, help='Number of iteration to be performed (Default: 5000).', default=5000)

parser.add_argument("-e", "--evaporation_rate", type=float, help='The rate in which pheromone will evaporate from the trail per iteration (Default:0.1). It controls the convergence.', default=0.1)

parser.add_argument("-j", "--jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

parser.add_argument("-k", "--k_top", type=int, help='the k best ants will deposit pheromone on trail (Default:10). If k=0 then all ants will deposit pheromone.', default=10)

parser.add_argument("-a", "--alpha", type=float, help='Controls the influence of the amount of pherome over the propability of moving to the next node (Default: 1).', default=1.0)

parser.add_argument("-b", "--beta", type=float, help='Controls the influence of the heuristic values over the propability of moving to the next node (Default: 1).', default=1.0)

parser.add_argument("--trials", type=int, help='Number of trials (Default:10).', default=10)

args = parser.parse_args()

weight_matrix, b_node, e_node = aco.load_graph_file('entradas/entrada1.txt')

colony = aco.AntColony(n_ants=args.n_ants, iterations=args.iterations, evaporation_rate=args.evaporation_rate, k=args.k_top)

best_ant = colony.meta_heuristic(weight_matrix, b_node, e_node)
print best_ant.path_, best_ant.path_length_

#ant = aco.Ant()

#colony.init_pheromone_matrix_(weight_matrix)
#ant.random_walk(colony.pheromone_matrix_, weight_matrix, b_node, e_node)