# This file does hyperparameter search to find optimal parameters for
# a particular demand environment. In real life, this algorithm should
# regularly be performed on all historical data to maximise future
# earnings.

from utils import load_problem_data
import json
import polars as pl
from polars import col as F
from line_profiler import profile
from naive import projected_fleet_profit, get_maintenance_cost

import logging, os, sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(getattr(logging, os.environ.get('TIME_MACHINE_LOG_LEVEL', 'DEBUG')))
logger = logging.getLogger("time_machine")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

from dataclasses import dataclass

@dataclass
class Parameters:
    dismiss_delta: int = 5
    method: str = 'delta'

Parameters.MATRIX = {
    'dismiss_delta' : [ 9, 10, 11 ],
    'method' : [ 'delta', 'omegasum' ]
}

@profile
def get_my_solution(
        demand,
        path,
        log,
        /,
        parameters=Parameters()
        
) -> list:
    server_generations = ["GPU.S1", "GPU.S2", "GPU.S3", "CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4"]
    latency_sensitivities = ["low", "high", "medium"]
    demand = pl.DataFrame(demand)
    _, datacenters, servers, selling_prices = [pl.DataFrame(df) for df in load_problem_data()]
    with open(path) as file:
        actions = pl.DataFrame(json.load(file))

    # just do the simple solution and check if there are any dogshit servers that could not possibly have made money in their time.
    # for server_id in ['Ag']:
    running_loss = 0
    IG_dmd = { G : v for [G], v in demand.group_by('server_generation') }
    demands_IG = { (I, G) : { d['time_step'] : d[I] for d in IG_dmd[G]['time_step', I].to_dicts() }
                    for I in latency_sensitivities
                    for G in server_generations }
    joined_actions = actions \
        .join(datacenters, on='datacenter_id') \
        .join(servers, on='server_generation') \
        .join(selling_prices, on=['server_generation', 'latency_sensitivity'])
    server_events = { server_id : rest.to_dicts() for [server_id], rest in joined_actions.group_by('server_id') }
    costs = { datacenter_id : v['latency_sensitivity', 'cost_of_energy'].to_dicts()[0] for [datacenter_id], v in datacenters.group_by('datacenter_id') }
    selling_prices_IG = { (d['latency_sensitivity'], d['server_generation']) : d['selling_price'] for d in selling_prices.to_dicts() }
    bad_servers = []
    
    capacity_TIG = {}
    capacity_IG = {}
    
    
    for server_id, events in server_events.items():
        last_I = None
        
        for event in events:
            T = event['time_step']
            I = event['latency_sensitivity']
            G = event['server_generation']
            
            if (I, G) not in capacity_IG:
                capacity_IG[I, G] = 0
            if event['action'] == 'buy':
                capacity_IG[(I, G)] += event['capacity']
            elif event['action'] == 'move':
                if (last_I, G) not in capacity_IG:
                    capacity_IG[last_I, G] = 0
                capacity_IG[(last_I, G)] -= event['capacity']
                capacity_IG[(I, G)] += event['capacity']
                capacity_TIG[(T, last_I, G)] = capacity_IG[(last_I, G)]
            elif event['action'] == 'dismiss':
                capacity_IG[(I, G)] -= event['capacity']

            capacity_TIG[(T, I, G)] = capacity_IG[(I, G)]
                
            last_I = I

    romulus = pl.DataFrame([dict(T=T, I=I, G=G, D=D) for [T, I, G], D in capacity_TIG.items()])
    
    for server_id, events in server_events.items():
        buy = events[0]
        L_at_purchase = log[buy['time_step'] - 1]['L']
        U_at_purchase = log[buy['time_step'] - 1]['U']
        LU_at_purchase = L_at_purchase * U_at_purchase 

        # if L < 0.25:
        #     # The down payment on young servers is not relevant
        #     # because L makes all cost essentially free
        #     continue
        
        moves = {}
        dismiss = { 'time_step' : 168 }
        for event in events[1:]:
            if event['action'] == 'move':
                moves[event['time_step']] = event
            elif event['action'] == 'dismiss':
                dismiss = event
        
        balance = - buy['purchase_price'] * LU_at_purchase
        capacity = buy['capacity']

        I = buy['latency_sensitivity']
        G = buy['server_generation']
        demands = demands_IG[(I, G)]
        
        t = buy['time_step']
        
        last_profit = None
        peak_balance = balance
        peak = buy['time_step']
        for k in range(buy['time_step'], dismiss['time_step'] + 1):
            D = demands.get(k, 0)
            C = capacity_TIG.get((k, I, G), 0)
            if C > D:
                SUP = D / C
            else:
                SUP = 1.0
            L, U = log[k - 1]['L'], log[k - 1]['U']
            LU = L * U
            if k in moves:
                datacenter = costs[moves[k]['datacenter_id']]
                buy['cost_of_energy'] = datacenter['cost_of_energy']
                buy['selling_price'] = selling_prices_IG[(datacenter['latency_sensitivity'], G)]
                moving_cost = buy['cost_of_moving']
            else:
                moving_cost = 0.0
            age = k - buy['time_step'] + 1
            # update to reflect cost of energy changing on moves
            energy_costs = buy["energy_consumption"] * buy['cost_of_energy']
            maintenance_costs = get_maintenance_cost(buy["average_maintenance_fee"], age + (k - t), buy["life_expectancy"])
            # assume optimal capacity
            revenue = capacity * buy["selling_price"]
            profit = revenue * SUP - energy_costs - maintenance_costs - moving_cost
            balance += profit * LU
            if balance > peak_balance:
                peak = k
                peak_balance = balance
            if profit > 0:
                last_profit = k
            # print(f'{k}: {int(balance):,} (ẟ{int(profit):,})')
        # if balance < - buy['purchase_price']:
        report = {
            **buy,
            'omega' : dismiss['time_step'],
            'last_profit': last_profit,
            'peak': peak,
            'balance' : balance,
            'L' : L_at_purchase,
            'U' : U_at_purchase
        }
        if balance < 0.0:
        # if green = False:
            running_loss += balance - buy['purchase_price']
            logger.debug(f'{server_id}: {int(balance):,} (running loss: {int(running_loss):,}, {last_profit} from {k})')
            # bad servers, so called without knowledge of the initial
            # L value which ennulls expenditure that then make money
            # in good times.
            bad_servers.append(report)

    
    slig = pl.DataFrame(bad_servers)
    bad_servers = slig.filter(F('omega') - F('time_step') < 80).to_dicts()
    
    bad_servers_0_set = { b['server_id'] : b for b in bad_servers }
    servers_to_skip = set()
    zumms = []
    
    for value in actions.to_dicts():
        if value['server_id'] in servers_to_skip:
            continue
        if value['server_id'] in bad_servers_0_set:
            bad_server = bad_servers_0_set[value['server_id']]
            lp = bad_server['last_profit']
            omega = bad_server['omega']
            peak = bad_server['peak']
            if parameters.method == 'delta':
                target_dismissal = value['time_step'] - parameters.dismiss_delta
            elif parameters.method == 'peak':
                target_dismissal = int(peak + omega) / 2
            else:
                target_dismissal = int (lp + omega / 2) if lp is not None else value['time_step'] - parameters.dismiss_delta
            if value['action'] == 'dismiss':
                value['time_step'] = target_dismissal
            elif value['time_step'] >= target_dismissal:
                if value['action'] == 'buy':
                    servers_to_skip.add(value['server_id'])
                # drop this event
                continue
        zumms.append(value)
    return zumms
