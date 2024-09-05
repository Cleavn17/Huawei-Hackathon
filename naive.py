from utils import load_problem_data
import numpy as np
import base64, struct
import json
import polars as pl
from polars import col as F
from functools import lru_cache
from line_profiler import profile
import logging
from sys import stdout
from os import environ

handler = logging.StreamHandler(stdout)
handler.setLevel(getattr(logging, environ.get('NAIVE_LOG_LEVEL', 'DEBUG')))

logger = logging.getLogger("naive")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def projected_fleet_profit(n, cost_of_energy, server, demands, t, break_even_per_server=0.0, initial_balance_per_server=0.0, lookahead=90, steps=1, all=False, ages=None):
    scenarios = []
    capacity = n * server["capacity"]
    if ages is None:
        ages = np.broadcast_to(1, n)
    for ratio in range(steps):
        scalar = (steps - ratio) / steps
        scaled_need = int(n * scalar)
        balance = initial_balance_per_server * scaled_need
        for k in range(t, min(t + lookahead, 168 + 1)):
            capacity_served = min(capacity, demands.get(k) or 0)
            energy_costs = server["energy_consumption"] * cost_of_energy
            maintenance_costs = get_maintenance_cost(server["average_maintenance_fee"], ages + (k - t), server["life_expectancy"]).sum()
            revenue = capacity_served * server["selling_price"]
            profit = revenue - scaled_need * energy_costs - maintenance_costs
            balance += profit
            # extrapolate to break-even and then some
        if balance > scaled_need * break_even_per_server or all:
            scenarios.append((scalar, balance))
    return scenarios

def compress(i):
    return base64.b64encode(struct.pack('<i', i).rstrip(b'\x00')).strip(b'=')

def get_maintenance_cost(b, x, xhat):
    # Copied from evaluation.py
    return b * (1 + (((1.5)*(x))/xhat * np.log2(((1.5)*(x))/xhat)))

@profile
def get_my_solution(demand) -> list:
    demand = pl.DataFrame(demand)
    _, datacenters, servers, selling_prices = [pl.DataFrame(df) for df in load_problem_data()]

    @lru_cache
    def get_server_with_selling_price(i, g) -> dict:
        server = servers.join(selling_prices.filter(F('latency_sensitivity') == i), on=['server_generation'], how="inner").filter(F('server_generation') == g).to_dicts()[0]
        server["release_start"], server["release_end"] = json.loads(server.pop("release_time"))
        return server
    
    i = 1
    actions = []
    server_generations = ["GPU.S1", "GPU.S2", "GPU.S3", "CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4"]
    # server_generations = ["CPU.S1"]
    latency_sensitivities = ["low", "high", "medium"]
    # latency_sensitivities = ["low"]

    stock_schema = {
        'time_step' : pl.Int64,
        'datacenter_id' : pl.String,
        'server_generation' : pl.String,
        'server_id' : pl.String,
        'action' : pl.String,
        'latency_sensitivity' : pl.String,
        'slots_size' : pl.Int64,
    }

    stock = pl.DataFrame(schema=stock_schema)
    
    DC_SERVERS = {}
    DC_SLOTS = {}
    DC_SCOPED_SERVERS = {}
    DC_DEAD_AT = {}

    @profile
    def expire_ids(ids, do_book_keeping=True):
        combination = set([*ids])
        drop_condition = F('server_id').is_in(combination)
        split = { k : v for [k], v in stock.group_by(drop_condition) }
        dropped, retained = split.get(True, pl.DataFrame(schema=stock_schema)), split.get(False, pl.DataFrame(schema=stock_schema))
        for server in dropped.to_dicts():
            actions.append({
                "time_step" : t,
                "datacenter_id" : server["datacenter_id"],
                "server_generation" : server["server_generation"],
                "server_id" : server['server_id'],
                "action" : "dismiss"
            })

            if do_book_keeping:
                DC_SERVERS[server["datacenter_id"]] -= 1
                DC_SLOTS[server["datacenter_id"]] -= server["slots_size"]
                DC_SCOPED_SERVERS[(server["datacenter_id"], server["server_generation"])] -= 1
        return retained

    old_expiry_list = []
    
    for t in range(168):
        t = t + 1

        dead_ids = DC_DEAD_AT.get(t, [])
        if len(dead_ids) > 0:
            stock = expire_ids(dead_ids)
    
        existing = { k : v for k, v in stock.group_by(['datacenter_id', 'server_generation']) }
        demand_profiles = []

        # Increases profits by about 10 million
        DBS = { k : v.to_dicts() for [k], v in datacenters.group_by('latency_sensitivity') }
        IG_existing_sum = { k : v['slots_size'].sum() for k, v in stock.group_by(['latency_sensitivity', 'server_generation']) }
        IG_dmd = { G : v for [G], v in demand.group_by('server_generation') }
        
        for I in latency_sensitivities:
            for G in server_generations:
                S = get_server_with_selling_price(I, G)
                E_ig = datacenters.filter(F('latency_sensitivity') == I)['cost_of_energy'].mean()
                capacity_to_offer = datacenters.filter(F('latency_sensitivity') == I)['slots_capacity'].sum() - IG_existing_sum.get((I, G), 0)
                n = capacity_to_offer // S['slots_size']
                demands = { d['time_step'] : d[I] for d in IG_dmd[G]['time_step', I].to_dicts() }
                [(_, profit)] = projected_fleet_profit(t=t, n=n, cost_of_energy=E_ig, server=S, demands=demands, all=True, lookahead=5)
                demand_profiles.append((I, G, profit))

        demand_profiles = sorted(demand_profiles, key=lambda p: -p[2])
        
        expiry_pool = {}
        
        expiry_list = []
        
        for I, G, _ in demand_profiles:
            # need to make more wholistic …
            for candidate in DBS[I]:
                datacenter_id = candidate['datacenter_id']
                
                D_ig_old = int(IG_dmd[G].filter(F('time_step').is_between(t - 1, t - 1 + 10))[I].mean() or 0)
                # D_ig_old = int(IG_dmd[G].filter(F('time_step') == t - 1)[I].mean() or 0)

                D_ig_next = int(IG_dmd[G].filter(F('time_step').is_between(t, t + 5))[I].max() or 0)
                # D_ig_next = int(IG_dmd[G].filter(F('time_step') == t + 1)[I].mean() or 0)

                D_ig = int(IG_dmd[G].filter(F('time_step').is_between(t, t + 10))[I].mean() or 0)
                # D_ig = int(IG_dmd[G].filter(F('time_step') == t)[I].mean() or 0)
                
                server = get_server_with_selling_price(I, G)
                servers_needed_to_meet_demand = D_ig // server['capacity']
                servers_needed_to_meet_demand_next = D_ig_next // server['capacity']
                servers_in_stock = DC_SCOPED_SERVERS.get((datacenter_id, G), 0)
                # servers_in_stock = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) for candidate in DBS[I])

                # We do this to prevent selling a chip in one time step only to need it again in the next timestep due to the random noise added to
                # the demand in the evaluation script. Though it's impact seems to be negligible
                excess = max(0, servers_in_stock - max(servers_needed_to_meet_demand, servers_needed_to_meet_demand_next))

                if G not in expiry_pool:
                    expiry_pool[G] = []
                
                if excess > 0:
                    demands = { d['time_step'] : d[I] for d in IG_dmd[G]['time_step', I].to_dicts() }
                    ages = stock.filter((F('server_generation') == G) & (F('datacenter_id') == datacenter_id)).with_columns(k=t - F('time_step'))['k']
                    W = projected_fleet_profit(t=t, n=servers_in_stock, server=server, demands=demands, cost_of_energy=candidate["cost_of_energy"], all=True, ages=ages, lookahead=10, steps=5)
                    W = sorted(W, key=lambda p: -p[1])
                    excess = int((excess + servers_in_stock - servers_in_stock * W[-1][0]) / 2)

                # if (utilisation < 0.9 or servers_needed_to_meet_demand == 0):
                    servers_to_merc = existing[(datacenter_id, G)].sort('time_step')[-excess:]
                    for server in servers_to_merc.to_dicts():
                        DC_SERVERS[server["datacenter_id"]] -= 1
                        DC_SLOTS[server["datacenter_id"]] -= server["slots_size"]
                        DC_SCOPED_SERVERS[(server["datacenter_id"], server["server_generation"])] -= 1
                    logger.debug(f"\tExpiring {len(servers_to_merc):5,} of {I}-{G} ({D_ig_old:,} 📉 {D_ig:,})")
                    expiry_list.append(f'{I}-{G}')
                    expiry_pool[G] += [*servers_to_merc['server_id']]

        for I, G, _ in demand_profiles:
            for candidate in DBS[I]:
                datacenter_id = candidate['datacenter_id']
                
                slots_capacity = candidate['slots_capacity']
                D_ig = int(IG_dmd[G].filter(F('time_step') == t)[I].mean() or 0)
                server = get_server_with_selling_price(I, G)
                slots_size = server['slots_size']
                servers_needed_to_meet_demand = D_ig // server['capacity'] # // len(candidates)
                slots_used_in_datacenter = DC_SLOTS.get(datacenter_id, 0)
                assert slots_used_in_datacenter <= slots_capacity, "stock must be <=capacity"
                servers_in_stock = DC_SCOPED_SERVERS.get((datacenter_id, G), 0)
                if servers_in_stock < servers_needed_to_meet_demand:
                    
                    capacity_remaining = slots_capacity - slots_used_in_datacenter
                    assert capacity_remaining >= 0, f"capacity remaining ({capacity_remaining}) ought to be >=0"
                    need = np.clip(servers_needed_to_meet_demand - servers_in_stock, 0, capacity_remaining // slots_size)
                    
                    # ASSUME MOVED SERVERS ACT AS FRESH SERVERS (THIS IS FALSE AND MAINTENANCE COST IS MUCH HIGHER, THERE STILL MAY BE NO BREAK EVEN)
                    pool = expiry_pool.get(G, [])
                    amount_to_take_from_pool = min(need, len(pool))
                    if amount_to_take_from_pool > 0:
                        taken = pool[-amount_to_take_from_pool:]
                        untaken = pool[:-amount_to_take_from_pool]
                        expiry_pool[G] = untaken
                        need -= len(taken)
                        moved = stock.filter(F('server_id').is_in(taken) & (F('datacenter_id') != datacenter_id))
                        stock = stock.with_columns(
                            datacenter_id=pl.when(F('server_id').is_in(taken)).then(pl.lit(datacenter_id)).otherwise('datacenter_id'),
                            latency_sensitivity=pl.when(F('server_id').is_in(taken)).then(pl.lit(I)).otherwise('latency_sensitivity')
                        )
                        
                        move_actions = []
                        for _server in moved.to_dicts():
                            move_actions.append({
                                "time_step" : t,
                                "datacenter_id" : datacenter_id,
                                "server_generation" : G,
                                "server_id" : _server['server_id'],
                                "action" : "move"
                            })

                        logger.debug(f"\tHarvisting {len(taken):,} of {I}-{G} from expiry pool for €{len(moved)*server['cost_of_moving']:,} (ẟ{len(taken)*server['capacity']:,} to meet {D_ig:,})")
                            
                        DC_SERVERS[datacenter_id] = DC_SERVERS.get(datacenter_id, 0) + len(taken)
                        DC_SLOTS[datacenter_id] = DC_SLOTS.get(datacenter_id, 0) + len(taken) * slots_size
                        DC_SCOPED_SERVERS[(datacenter_id, G)] = DC_SCOPED_SERVERS.get((datacenter_id, G), 0) + len(taken)

                        actions += move_actions

                    # NEED TO AUGMENT SIMULATED PROFITABILITY PLANNING

                    if need > 0:
                        P = server["purchase_price"]
                        existing_capacity = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) * server["capacity"] for candidate in DBS[I])
                        # Assume that when the demand falls below the existing capacity, no capacity will go to the new servers
                        demands = { d['time_step'] : max(0, d[I] - existing_capacity)  for d in IG_dmd[G]['time_step', I].to_dicts() }
                        profitable_scenarios = projected_fleet_profit(
                            n=need,
                            cost_of_energy=candidate["cost_of_energy"],
                            server=server,
                            demands=demands,
                            t=t,
                            break_even_per_server=P/5,
                            initial_balance_per_server=-P
                        )
                        is_profitable_scenario = len(profitable_scenarios) != 0
                        profitable_scenarios = sorted(profitable_scenarios, key=lambda m: m[1])
                        if is_profitable_scenario and len(profitable_scenarios):
                            best_scale, _ = profitable_scenarios[-1]
                            formatted = [f'{scalar} €{int(profit_earned):,}' for scalar, profit_earned in profitable_scenarios]
                            logger.debug(f"\tBest choice is: '{formatted[-1]}', others: {formatted[:-1]}")
                            need = int(need * best_scale)
                            logger.debug(f"\tBuy {need} more {I}-{G} servers for €{int(need*P):,}!")
                        else:
                            logger.debug(f"\tDon't drop €{int(need*P):,} on {need} more {I}-{G} servers! ")

                        if t >= server["release_start"] and t <= server["release_end"] and is_profitable_scenario:
                            if f'{I}-{G}' in old_expiry_list:
                                warning = f"([38;2;255;0;0m🤡 {I}-{G} was expired last round! 🤡[m)"
                            else:
                                warning = ""
                            logger.debug(f"\tPurchasing {need:,} of {I}-{G} for €{int(need * server["purchase_price"]):,} (ẟ{need * server["capacity"]:,} to meet {D_ig:,}) {warning}")

                            buy_actions = [
                                {
                                    "time_step" : t,
                                    "datacenter_id" : datacenter_id,
                                    "server_generation" : G,
                                    "server_id" : compress(i := i + 1).decode(),
                                    "action" : "buy"
                                }
                                for _ in range(need)
                            ]
                            actions += buy_actions
                            bought = pl.DataFrame([{**action, "latency_sensitivity" : I, "slots_size" : slots_size } for action in buy_actions], schema=stock_schema)
                            stock = pl.concat([stock, bought])

                            DC_SERVERS[datacenter_id] = DC_SERVERS.get(datacenter_id, 0) + len(bought)
                            DC_SLOTS[datacenter_id] = DC_SLOTS.get(datacenter_id, 0) + len(bought) * slots_size
                            DC_SCOPED_SERVERS[(datacenter_id, G)] = DC_SCOPED_SERVERS.get((datacenter_id, G), 0) + len(bought)

                            delta = 5
                            DC_DEAD_AT[t + server['life_expectancy'] - delta] = DC_DEAD_AT.get(t + server['life_expectancy'] - delta, []) + [server['server_id'] for server in buy_actions]

        excess_ids = []
        for G in server_generations:
            excess_ids += expiry_pool.get(G, [])
        if len(excess_ids) > 0:
            stock = expire_ids(excess_ids, do_book_keeping=False)

        K = pl.concat([demand.filter(F('time_step') == t).select('server_generation', I).rename({ I : 'demand' }).with_columns(latency_sensitivity=pl.lit(I)) for I in latency_sensitivities])
        E = stock.join(servers, on="server_generation", how="inner") \
            .join(datacenters, on='datacenter_id') \
            .with_columns(energy=F('energy_consumption') * F('cost_of_energy'))['energy'].sum() or 0
        R = stock.join(servers, on="server_generation", how="inner") \
            .join(selling_prices, on=["latency_sensitivity", "server_generation"], how="inner") \
            .join(datacenters, on='datacenter_id') \
            .group_by(["latency_sensitivity", "server_generation"]) \
            .agg(F('capacity').sum(), F('selling_price').mean()) \
            .join(K, on=["server_generation", "latency_sensitivity"], how="inner") \
            .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') * F('selling_price'))['capacity'].sum() or 0
        C = stock.join(servers, on="server_generation", how="inner").filter(F('time_step') == t)['purchase_price'].sum() or 0
        P = (R - C - E) or 0
        L = stock.join(servers, on="server_generation", how="inner").select((t - F('time_step') + 1) / F('life_expectancy')).mean().item() or 0.0
        U = stock.join(servers, on="server_generation", how="inner") \
            .select("latency_sensitivity", 'server_generation', 'capacity') \
            .group_by(["latency_sensitivity", "server_generation"]) \
            .agg(F('capacity').sum()) \
            .join(K, on=["server_generation", "latency_sensitivity"], how="inner") \
            .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') / F('capacity'))['capacity'].mean() or 0.0
        O = P * L * U
        logger.debug(f"{t:3}: O: {int(O):<11,}, P: {int(P):<11,} (R={int(R):<11,}, C={int(C):11,},  E={int(E):11,}), L: {L:3.02}, U: {U:3.02}")

        slots = { slot['datacenter_id'] : slot['slots_size'] for slot in stock.group_by('datacenter_id').agg(F('slots_size').sum()).to_dicts() }
        _datacenters = { datacenter['datacenter_id'] : datacenter['slots_capacity'] for datacenter in datacenters.to_dicts() }
        logger.info(f"{t:3}: %s", ", ".join(f"DC{i + 1}: {slots.get(f'DC{i + 1}', 0):6} ({int(100*(slots.get(f'DC{i + 1}', 0)/_datacenters[f'DC{i + 1}'])):3}%)" for i in range(4)))

        # make equivalents which make caches redundant (please god keep this up to date…)
        db_scoped_servers = { k : len(v) for k, v in stock.group_by(['datacenter_id', 'server_generation']) }
        db_servers = { k : len(v) for [k], v in stock.group_by('datacenter_id') }
        db_slots = { k : v['slots_size'].sum() for [k], v in stock.group_by('datacenter_id') }
        for id in _datacenters.keys():
            assert stock.filter((F('datacenter_id') == id))['slots_size'].sum() <= datacenters.filter(F('datacenter_id') == id)['slots_capacity'].item(), "Constraint 2 violated"
            db = db_servers.get(id, 0)
            c = DC_SERVERS.get(id, 0)
            assert c == db, f"Database ({db}) and DC_SERVERS ({c}) are out of sync for {id}"
            assert DC_SLOTS.get(id, 0) == db_slots.get(id, 0), f"Database and DC_SLOTS are out of sync for {id}"
            for G in server_generations:
                assert DC_SCOPED_SERVERS.get((id, G), 0) == db_scoped_servers.get((id, G), 0), f"Database and DC_SCOPED_SERVERS are out of sync for {id}"

        old_expiry_list = expiry_list

    return actions

if __name__ == "__main__":
    demand, *_ = load_problem_data()
    solution = get_my_solution(demand)
    with open ("kjv.json", "w") as w:
        json.dump(solution, w, indent=1)
