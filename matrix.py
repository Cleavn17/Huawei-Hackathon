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
parser.add_argument('--limit', type=int, default=168)
args = parser.parse_args()

seeds = [args.seed] if args.seed is not None else known_seeds('test')

demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

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
        
        solution = fleet, pricing_strategy = bases.get(N_j) or get_my_solution(actual_demand, parameters=N, limit=args.limit)
        fleet, pricing_strategy = pd.DataFrame(fleet), pd.DataFrame(pricing_strategy)
        bases[N_j] = solution
        
        solution_path = f'/tmp/{code}-unmutated.json'
        save_solution(fleet, pricing_strategy, solution_path)
        
        score = bases_E[N_j] if N_j in bases_E else evaluation_function(*load_solution(solution_path), demand, datacenters, servers, selling_prices, elasticity, seed=seed, verbose=args.verbose, actual_demand=actual_demand, time_steps=args.limit)
        bases_E[N_j] = score

        mutated_score = 0.0
        mutated_solution = solution
        
        # mutated_solution = mutated_fleet, mutated_pricing_strategy = mods.get((N_j, T_j)) or mutate(actual_demand, solution_path)
        # mutated_fleet, mutated_pricing_strategy = pd.DataFrame(mutated_fleet), pd.DataFrame(mutated_pricing_strategy)
        # mods[(N_j, T_j)] = mutated_solution
        # 
        # mutated_solution_path = f'/tmp/{code}-mutated.json'
        # save_solution(mutated_fleet, mutated_pricing_strategy, mutated_solution_path)
        # 
        # mutated_score = mods_E[(N_j, T_j)] if (N_j, T_j) in mods_E else evaluation_function(*load_solution(mutated_solution_path), demand, datacenters, servers, selling_prices, elasticity, seed=seed, verbose=args.verbose, actual_demand=actual_demand, time_steps=args.limit)
        # mods_E[(N_j, T_j)] = mutated_score
        #
        
        report = {
            'seed' : seed,
            'code' : code,
            'path' : solution_path,
            'score' : score,
            'mutated_score' : mutated_score
        }
        
        with open (f'/tmp/{code}-report.json', 'w') as f:
            json.dump(report, f)
            
        scores.append(report)

        if high_score is None or score > high_score or mutated_score > high_score:
            report_path = join(args.session, "reports", f'{seed}.json')
            solution_path = join(args.session, "solutions", f'{seed}.json')
            maybe_high_score = max(high_score or 0.0, mutated_score, score)
            
            if Path(report_path).exists():
                with open(report_path) as f: existing_report = json.load(f)
                
                existing_high_score = max(existing_report['score'], existing_report['mutated_score'])
                if existing_high_score > maybe_high_score:
                    print(f'\t⚠ Not writing as actual high score for {seed} is {existing_high_score} ({existing_report["code"]})')
                    high_score = existing_high_score
                    high_score_code = existing_report['code']
                    continue
            
            if high_score is None or score > high_score:
                print(f"Reached new high score !!!")
                with open(solution_path, 'w') as f: json.dump(solution, f)
                with open(report_path, 'w') as f: json.dump(report, f)

            if high_score is None or mutated_score > high_score:
                print(f"Reached new high score (MUTANT) !!!")
                with open(solution_path, 'w') as f: save_solution(fleet, solution, f)
                with open(report_path, 'w') as f: json.dump(report, f)

            high_score = maybe_high_score
            high_score_code = code

            print(f"\tseed: {seed}, score: {int(high_score):,}, code: {high_score_code}")

        else:
            print(f'Got ({int(score):,}, {int(mutated_score):,}) for {code}')

