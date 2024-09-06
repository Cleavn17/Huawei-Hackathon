from utils import (load_problem_data, load_solution)
from evaluation import evaluation_function

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('file', default='./data/solution_example.json')
parser.add_argument('--seed', default=2237, type=int)
parser.add_argument('--silent', const=False, dest="verbose", action="store_const", default=True)
args = parser.parse_args()

seed = args.seed

print(f'Evaluating seed {seed}')

# LOAD SOLUTION
solution = load_solution(args.file)

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# EVALUATE THE SOLUTION
score, _ = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=seed, verbose=args.verbose, return_objective_log=True)

print(f'Solution score (seed={seed}): {score}')
