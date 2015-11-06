import time
import md5
import argparse
import numpy as np
import AntColonyOptimization as aco
import pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Heuristic to solve the longest path problem based on Ant Colony Optimization(ACO).")

	parser.add_argument("input", type=str,
	                    help="Input file")

	parser.add_argument("-n", "--n_ants", type=int, help='Number of artificial ants per itereation (default: 200)', default=200)

	parser.add_argument("-i", "--iterations", type=int, help='Number of iteration to be performed (Default: 5000).', default=5000)

	parser.add_argument("-e", "--evaporation_rate", type=float, help='The rate in which pheromone will evaporate from the trail per iteration (Default:0.1). It controls the convergence.', default=0.1)

	parser.add_argument("-j", "--jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

	parser.add_argument("-k", "--k_top", type=int, help='the k best ants will deposit pheromone on trail (Default:10). If k=0 then all ants will deposit pheromone.', default=10)

	parser.add_argument("-a", "--alpha", type=float, help='Controls the influence of the amount of pherome over the propability of moving to the next node (Default: 1).', default=1.0)

	parser.add_argument("-b", "--beta", type=float, help='Controls the influence of the heuristic values over the propability of moving to the next node (Default: 1).', default=1.0)

	parser.add_argument("--trials", type=int, help='Number of trials (Default:30).', default=30)

	parser.add_argument("-s","--serialize", action="store_true", help='Serialize and save in file the solutions.')

	parser.add_argument("-d", "--dump_folder", type=str, help='Logger\'s dump folder (Default: ./).', default=".")

	args = parser.parse_args()

	weight_matrix, b_node, e_node = aco.load_graph_file(args.input)

	output = md5.new(str(vars(args))).hexdigest()[:5] + "-" + md5.new(str(time.gmtime())).hexdigest()[:5] + ".out"

	colony = aco.AntColony(base_ant=aco.Ant(alpha=args.alpha,beta=args.beta),n_ants=args.n_ants, iterations=args.iterations, evaporation_rate=args.evaporation_rate, k=args.k_top, n_jobs=args.jobs, logger=aco.Logger(args.dump_folder + "/" + output, append=True), random_state=42)

	colony.logger.running_setup(vars(args))

	ants = []
	lengths = np.ndarray(args.trials)
	for t in range(args.trials):
		best_ant = colony.meta_heuristic(weight_matrix, b_node, e_node)
		ants.append(best_ant)
		lengths[t] = best_ant.path_length_
		#"Trial #%d - Length : %d" % 
		print (t + 1), best_ant.path_length_

	if args.serialize:
		pickle.dump(ants, open(args.dump_folder + "/" + output + "_solutions", 'w'))

	print colony.logger.running_summary()