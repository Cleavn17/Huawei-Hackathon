from utils import (load_problem_data, load_solution)
from evaluation import evaluation_function
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file', default='./data/solution_example.json')
args = parser.parse_args()

# LOAD SOLUTION
solution = load_solution(args.file)

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=122)

print(f'Solution score: {score}')
