
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from naive import get_my_solution

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seed', required=False, type=int)
parser.add_argument('--mutate', required=False)
parser.add_argument('--eval', const=True, dest="eval", action="store_const", default=False)
args = parser.parse_args()
seeds = [args.seed] if args.seed is not None else known_seeds()

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    fleet, pricing_strategy = get_my_solution(actual_demand)

    fleet, pricing_strategy = pd.DataFrame(fleet), pd.DataFrame(pricing_strategy)

    # SAVE YOUR SOLUTION
    save_solution(fleet, pricing_strategy, f'./output/{seed}.json')

