# Longest path problem heuristic based on Ant Colony Optimization (ACO)

This code was developed as part of the course Natural Computing of the Graduate Program in Computer Science of the Federal University of Minas Gerais.

## Code
The code can be easily extended to other problems. The class Ant holds the pheromone update rule and the probabilistic rule to choose the next node. Therefore, the class Ant can be extended to fulfill the scope of another problem. One can also extend the class AntColony for more specific changes.

## References

- [1] M. Dorigo, V. Maniezzo, and A. Colorni. The ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics–Part, 26:1–13, 1996.
- [2] M. R. Garey and D. S. Johnson. Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman & Co., New York, NY, USA, 1979.
- [3] W.-S. Yang and S.-X. Weng. Application of the ant colony optimization algorithm to the influence-maximization problem. International Journal of Swarm Intelligence and Evolutionary Computation, 2012.

## Usage

```
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
```
