from utils import (load_problem_data, load_solution)
from evaluation import evaluation_function
import numpy as np
import pandas as pd
from functools import cache
import random
import json
import itertools
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from naive import get_my_solution, Parameters as NaiveParameters
from timemachine import get_my_solution as mutate, Parameters as TimeMachineParameters
from pathlib import Path
from os.path import  join

def expand_matrix(Bearer):
    return [ Bearer(**{ k : v for k, v in zip(Bearer.MATRIX.keys(), combination) }) for combination in  itertools.product(*Bearer.MATRIX.values()) ]

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seed', required=False, type=int)
parser.add_argument('--mutate', required=False)
parser.add_argument('--silent', const=False, dest="verbose", action="store_const", default=True)
parser.add_argument('--session', default="session")
args = parser.parse_args()

seeds = [args.seed] if args.seed is not None else known_seeds('test')

demand, datacenters, servers, selling_prices = load_problem_data()

for seed in seeds:
    print(f"RUNNING WITH SEED {seed}")
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    naive_matrix = expand_matrix(NaiveParameters)
    random.shuffle(naive_matrix)
    time_machine_matrix = expand_matrix(TimeMachineParameters)
    random.shuffle(time_machine_matrix)
    
    scores = []
    
    high_score_code = None
    high_score = None

    bases = {}
    bases_E = {}
    mods = {}
    mods_E = {}
    
    combos = [*itertools.product(naive_matrix, time_machine_matrix)]
    random.shuffle(combos)

    Path(join(args.session, "solutions")).mkdir(parents=True, exist_ok=True)
    Path(join(args.session, "reports")).mkdir(parents=True, exist_ok=True)
    
    for N, T in combos:
        N_j = json.dumps(N.__dict__)
        T_j = json.dumps(T.__dict__)
        code = f'{N_j} | {T_j} | {seed}'
        print(code)
        
        solution = bases.get(N_j) or get_my_solution(actual_demand, parameters=N) ; bases[N_j] = solution
        solution_path = f'/tmp/{code}-unmutated.json'
        with open(solution_path, 'w') as f: json.dump(solution, f)
        (score, log) = bases_E[N_j] if N_j in bases_E else evaluation_function(load_solution(solution_path), demand, datacenters, servers, selling_prices, seed=seed, verbose=args.verbose, actual_demand=actual_demand, return_objective_log=True) ; bases_E[N_j] = (score, log)
        
        mutated_solution = mods.get((N_j, T_j)) or mutate(actual_demand, solution_path, log) ; mods[(N_j, T_j)] = mutated_solution
        mutated_solution_path = f'/tmp/{code}-mutated.json'
        with open(mutated_solution_path, 'w') as f: json.dump(mutated_solution, f)
        mutated_score = mods_E[(N_j, T_j)] if (N_j, T_j) in mods_E else evaluation_function(load_solution(mutated_solution_path), demand, datacenters, servers, selling_prices, seed=seed, verbose=args.verbose, actual_demand=actual_demand) ; mods_E[(N_j, T_j)] = mutated_score

        print(f'Got ({int(score):,}, {int(mutated_score):,}) for {code}')
        report = {
            'seed' : seed,
            'code' : code,
            'path' : mutated_solution_path,
            'score' : score,
            'mutated_score' : mutated_score
        }
        with open (f'/tmp/{code}-report.json', 'w') as f: json.dump(report, f)
        scores.append(report)
        
        if high_score is None or score > high_score:
            high_score = score
            high_score_code = code
            print(f"REACHED NEW HIGH SCORE !!! {score} with {high_score_code}")
            with open(join(args.session, "solutions", f'{seed}.json'), 'w') as f: json.dump(solution, f)
            with open(join(args.session, "reports", f'{seed}.json'), 'w') as f: json.dump(report, f)
        
        if high_score is None or mutated_score > high_score:
            high_score = mutated_score
            high_score_code = code
            print(f"REACHED NEW HIGH SCORE (MUTANT) !!! {high_score} with {high_score_code}")
            with open(join(args.session, "solutions", f'{seed}.json'), 'w') as f: json.dump(mutated_solution, f)
            with open(join(args.session, "reports", f'{seed}.json'), 'w') as f: json.dump(report, f)

