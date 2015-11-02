usage: main.py [-h] [-n N_ANTS] [-i ITERATIONS] [-e EVAPORATION_RATE]
               [-j JOBS] [-k K_TOP] [--trials TRIALS] [-s] [-d DUMP_FOLDER]
               input

Heuristic to solve the longest path problem based on Ant Colony
Optimization(ACO).

positional arguments:
  input                 Input file

optional arguments:
  -h, --help            show this help message and exit
  -n N_ANTS, --n_ants N_ANTS
                        Number of artificial ants per itereation (default:
                        200)
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iteration to be performed (Default: 5000).
  -e EVAPORATION_RATE, --evaporation_rate EVAPORATION_RATE
                        The rate in which pheromone will evaporate from the
                        trail per iteration (Default:0.1). It controls the
                        convergence.
  -j JOBS, --jobs JOBS  Number of CPUs available to parallelize the execution
                        (Default:1). If -1 is given then it gets all CPUs
                        available
  -k K_TOP, --k_top K_TOP
                        the k best ants will deposit pheromone on trail
                        (Default:10). If k=0 then all ants will deposit
                        pheromone.
  --trials TRIALS       Number of trials (Default:30).
  -s, --serialize       Serialize and save in file the solutions.
  -d DUMP_FOLDER, --dump_folder DUMP_FOLDER
                        Logger's dump folder (Default: ./).
