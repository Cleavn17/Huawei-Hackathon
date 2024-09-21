

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('file', default='./data/solution_example.json')
parser.add_argument('--seed', default=2237, type=int)
parser.add_argument('--silent', const=False, dest="verbose", action="store_const", default=True)
args = parser.parse_args()

seed = args.seed

# LOAD SOLUTION
fleet, pricing_strategy = load_solution(args.file)

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(fleet, pricing_strategy, demand, datacenters, servers, selling_prices, elasticity, seed=seed, verbose=args.verbose)

print(f'Solution score: {score}')



