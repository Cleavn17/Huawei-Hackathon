
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from naive import get_my_solution

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seed', required=False, type=int)
args = parser.parse_args()

seeds = [args.seed] if args.seed is not None else known_seeds('test')

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    print(f"RUNNING WITH SEED {seed}")
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

