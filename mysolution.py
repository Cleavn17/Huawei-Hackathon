
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution, load_problem_data
from evaluation import get_actual_demand
from naive import get_my_solution
from evaluation import evaluation_function

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seed', required=False, type=int)
parser.add_argument('--mutate', required=False)
parser.add_argument('--eval', const=True, dest="eval", action="store_const", default=False)
parser.add_argument('--verbose', const=True, dest="verbose", action="store_const", default=False)
parser.add_argument('--limit', type=int, dest="limit", default=168)
args = parser.parse_args()

limit = args.limit

seeds = [args.seed] if args.seed is not None else known_seeds()

demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    fleet, pricing_strategy = get_my_solution(actual_demand, limit=limit)
    fleet, pricing_strategy = pd.DataFrame(fleet), pd.DataFrame(pricing_strategy)

    if args.eval:
        score = evaluation_function(fleet, pricing_strategy, demand, datacenters, servers, selling_prices, elasticity, seed=seed, time_steps=limit)    
        print(f"Got: {score}")
    
    # SAVE YOUR SOLUTION
    save_solution(fleet, pricing_strategy, f'./output/{seed}.json')

