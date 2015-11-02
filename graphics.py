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


	args = parser.parse_args()

	out = aco.Displayer()
	#input = "results/12c1e778ed16979c7124fc03b3ed9db3"
	print out.format(args.input)