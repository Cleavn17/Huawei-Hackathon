from evaluation import evaluation_function
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution, load_problem_data, load_solution
from evaluation import get_actual_demand
from naive import get_my_solution
from timemachine import get_my_solution as mutate

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seed', required=False, type=int)
parser.add_argument('--mutate', required=False)
parser.add_argument('--eval', const=True, dest="eval", action="store_const", default=False)
args = parser.parse_args()

seeds = [args.seed] if args.seed is not None else known_seeds('test')

demand, datacenters, servers, selling_prices = load_problem_data()

for seed in seeds:
    print(f"RUNNING WITH SEED {seed}")
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)
    
    # SAVE YOUR SOLUTION
    if args.mutate is None:
        solution = get_my_solution(actual_demand)
        save_solution(solution, f'./output/{seed}.json')
        # save_solution(stocks, f'./output/{seed}-stocks.json')
        if args.eval:
            score, log = evaluation_function(load_solution(f'./output/{seed}.json'), demand, datacenters, servers, selling_prices, seed=seed, actual_demand=actual_demand, return_objective_log=True)
            print(f"Got: {score}")
            
    else:
        existing_solution_path = f'./output/{seed}.json'
        # with open(f'./output/{seed}-stocks.json') as f: stocks = json.loads(f)
        _, log = evaluation_function(load_solution(existing_solution_path), demand, datacenters, servers, selling_prices, seed=seed, actual_demand=actual_demand, return_objective_log=True)
        solution = mutate(actual_demand, existing_solution_path, log, None)
        save_solution(solution, f'./output/{seed}-mutate.json')
        
