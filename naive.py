# caith do dhochas agus do ghui sa bhruascar
# nil aon usaid acu
# is ifreann te gan reasun
# an cod faoi bhun

import itertools
import math
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

def get_demand_segments(complex=True):
    if complex:
        return (["GPU.S1", "GPU.S2", "GPU.S3", "CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4"], ["low", "high", "medium"])
    else:
        return (["CPU.S1"], ["low"])

# Takes an array servers considers the cost of energy and maintentance
# of each server and also considers the selling price of the services
# that the server can provide. It then extrapolates into the future
# using predicted demand and the selling price to determine how much
# profit the server will make in the future.

# This function is actually broken now since selling price can change
# at every time step but this function assumes that it doesn't change.

@profile
def projected_fleet_profit(
        # The number of servers to consider the profit for
        n,
        cost_of_energy,
        server, demands,
        t,
        break_even_per_server=0.0,
        initial_balance_per_server=0.0,
        lookahead=90,
        ratios=[1.0],
        all=False,
        ages=None,
        use_correct_capacity_formulation=False
):
    scenarios = []
    capacity = n * server["capacity"]
    for ratio in ratios:
        scaled_need = int(n * ratio)
        # below is the correct formulation but if fucks up other parts of the code for unknown reason. My solution leverages delusion I guess
        if use_correct_capacity_formulation:
            capacity = scaled_need * server["capacity"]
        balance = initial_balance_per_server * scaled_need
        for k in range(t, min(t + lookahead, 168 + 1)):
            capacity_served = min(capacity, demands.get(k) or 0)
            energy_costs = server["energy_consumption"] * cost_of_energy
            if ages is None:
                maintenance_costs = get_maintenance_cost(server["average_maintenance_fee"], k - t + 1, server["life_expectancy"]) * scaled_need
            else:
                maintenance_costs = get_maintenance_cost(server["average_maintenance_fee"], ages + (k - t), server["life_expectancy"]).sum()
            revenue = capacity_served * server["selling_price"]
            profit = revenue - scaled_need * energy_costs - maintenance_costs
            balance += profit
            # extrapolate to break-even and then some
        if balance > scaled_need * break_even_per_server or all:
            scenarios.append((ratio, balance))
    return scenarios

def get_maintenance_cost(b, x, xhat):
    # Copied from evaluation.py
    return b * (1 + (((1.5)*(x))/xhat * np.log2(((1.5)*(x))/xhat)))

from dataclasses import dataclass

@dataclass
class Parameters:
    expiry_lookahead: int = 11
    break_even_coefficient: float = 1/3
    reap_delta: int = 2

Parameters.MATRIX = {
    'expiry_lookahead' : [ 8, 9, 10, 11, 12 ],
    'break_even_coefficient' : [ -1/4, 0, 1/4, 1/3 ],
    # 'reap_delta' : [ 3, 5, 7 ]
    'reap_delta' : [ 2, 3, 4, 5 ]
}

@profile
def get_my_solution(
        demand,
        assertions_enabled = True,
        log_info = False,
        parameters = Parameters(),
        return_stock_log = False,
        limit=168
) -> list:
    demand, datacenters, servers, selling_prices, elasticity = [pl.DataFrame(df) for df in (demand, *load_problem_data()[1:])]

    elasticity_IG = { ig: e for ig, e in elasticity.group_by(['latency_sensitivity', 'server_generation']) }
    
    # "BROKEN" BY DYNAMIC SELLING PRICES
    def get_server_with_selling_price(i, g) -> dict:
        server = servers.join(selling_prices.filter(F('latency_sensitivity') == i), on=['server_generation']).filter(F('server_generation') == g).to_dicts()[0]
        server["release_start"], server["release_end"] = json.loads(server.pop("release_time"))
        return server
    
    current_server_index = 1
    actions = []
    trace = []

    server_generations, latency_sensitivities = get_demand_segments()

    def create_pricing_strategy(i: str, g: str, p: float, t: int):
        return {
            "time_step" : t,
            "latency_sensitivity" : i,
            "server_generation" : g,
            "price": p
        }

    def get_default_pricing_strategy_for_demand_segment(i, g, /, t):
        return create_pricing_strategy(i, g, get_server_with_selling_price(i, g)["selling_price"], t)
    
    default_pricing_strategy = [
        get_default_pricing_strategy_for_demand_segment(i, g, t=1)
        for i, g in itertools.product(latency_sensitivities, server_generations)
    ]

    pricing_strategy = [*default_pricing_strategy]

    stock_schema = {
        'time_step' : pl.Int64,
        'datacenter_id' : pl.String,
        'server_generation' : pl.String,
        'server_id' : pl.String,
        'action' : pl.String,
        'latency_sensitivity' : pl.String,
        'slots_size' : pl.Int64,
    }

    stocks = []
    balance_sheets = []

    stock = pl.DataFrame(schema=stock_schema)
    
    DC_SERVERS = {}
    DC_SLOTS = {}
    DC_SCOPED_SERVERS = {}
    dc_servers_to_delete_at = {}

    @profile
    def expire_ids(ids, do_book_keeping=True):
        dropped = { k : v for [k], v in stock.group_by(F('server_id').is_in(ids)) }
        if True in dropped:
            for server in dropped[True].to_dicts():
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

        return dropped[False] if False in dropped else pl.DataFrame(schema=stock_schema)

    old_expiry_list = []

    DBS_raw = { I : v for [I], v in datacenters.group_by('latency_sensitivity') }
    DBS = { I : v.to_dicts() for [I], v in datacenters.group_by('latency_sensitivity') }
    IG_base_demand = { G : v for [G], v in demand.group_by('server_generation') }
    
    for t in range(limit):
        t = t + 1
        step_trace = {}

        # When we purchase servers, we add a mark at the current
        # timestap plus the life expectancy of the server to indicate
        # that the server should be dismissed at that
        # point. Technically the evaluation software will dismiss
        # these servers automatically but manually dismissing servers
        # has some benifits. If we dismiss servers that we don't have,
        # the evaluation script will notify us of our mistake. We also
        # have to consider servers that are dismissed anyway because
        # we need to make sure that our stock is accurate.
        
        dead_ids = dc_servers_to_delete_at.get(t, [])
        if len(dead_ids) > 0:
            stock = expire_ids(dead_ids)
    
        existing = { k : v for k, v in stock.group_by(['datacenter_id', 'server_generation']) }
        
        IG_existing_sum = { k : v['slots_size'].sum() for k, v in stock.group_by(['latency_sensitivity', 'server_generation']) }
        demand_profiles = []

        # basically used as a way of taking an I-G pair and finding
        # the demand over a certain amount of timesteps. E.g. doing
        # averages and what not.
        
        
        for I, G in itertools.product(latency_sensitivities, server_generations):
            S = get_server_with_selling_price(I, G)
            E_ig = DBS_raw[I]['cost_of_energy'].mean()
            capacity_to_offer = sum(candidate['slots_capacity'] for candidate in DBS[I]) - IG_existing_sum.get((I, G), 0)
            n = capacity_to_offer // S['slots_size']
            demands = { d['time_step'] : d[I] for d in IG_base_demand[G]['time_step', I].to_dicts() }
            [(_, profit)] = projected_fleet_profit(t=t, n=n, cost_of_energy=E_ig, server=S, demands=demands, all=True, lookahead=40)
            demand_profiles.append((I, G, profit))

        demand_profiles = sorted(demand_profiles, key=lambda p: -p[2])
        expiry_pool = {}
        expiry_list = []
        ages_DG = { k : v.with_columns(k=t - F('time_step'))['k'] for k, v in existing.items() }
        
        for I, G, _ in demand_profiles:
            server = get_server_with_selling_price(I, G)
            global_servers_in_stock = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) for candidate in DBS[I])
            
            D_ig_real = int(IG_base_demand[G].filter(F('time_step') == t)[I].mean() or 0)
            if D_ig_real >= global_servers_in_stock * server["capacity"]:
                # we are well saturated, no need to expire anything
                continue
            
            # need to make more wholistic …
            for candidate in DBS[I]:
                datacenter_id = candidate['datacenter_id']
                servers_in_stock = DC_SCOPED_SERVERS.get((datacenter_id, G), 0)

                if servers_in_stock == 0:
                    # nothing to expire here
                    continue

                # If the demand saturates our servers, ignore the fact that the future may be bleak and make hay while the sun shines
                demands = { d['time_step'] : d[I] for d in IG_base_demand[G]['time_step', I].to_dicts() }
                ages = ages_DG[(datacenter_id, G)]

                D_ig_old       = int(IG_base_demand[G].filter(F('time_step').is_between(t - 1, t - 1 + parameters.expiry_lookahead))[I].mean() or 0)
                D_ig           = int(IG_base_demand[G].filter(F('time_step').is_between(t,     t     + parameters.expiry_lookahead))[I].mean() or 0)
                D_ig_next      = int(IG_base_demand[G].filter(F('time_step').is_between(t + 1, t + 1 + parameters.expiry_lookahead))[I].max() or 0)
                D_ig_next_next = int(IG_base_demand[G].filter(F('time_step').is_between(t + 2, t + 2 + parameters.expiry_lookahead))[I].max() or 0)
                
                servers_needed_to_meet_demand = D_ig // server['capacity']
                servers_needed_to_meet_demand_next = D_ig_next // server['capacity']
                servers_needed_to_meet_demand_next_next = D_ig_next_next // server['capacity']
                
                # We do this to prevent selling a chip in one time step only to need it again in the next timestep due to the random noise added to
                # the demand in the evaluation script. Though it's impact seems to be negligible
                
                excess = max(0, servers_in_stock - max(servers_needed_to_meet_demand, servers_needed_to_meet_demand_next, servers_needed_to_meet_demand_next_next))

                if G not in expiry_pool:
                    expiry_pool[G] = []
                
                if excess > 0:
                    W = projected_fleet_profit(t=t, n=servers_in_stock, server=server, demands=demands, cost_of_energy=candidate["cost_of_energy"], all=True, ages=ages, lookahead=10, ratios=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
                    W = sorted(W, key=lambda p: -p[1])
                    excess_W = servers_in_stock - servers_in_stock * W[-1][0]
                    alpha = 0.5
                    excess = int(alpha * excess + (1 - alpha) * excess_W)

                    servers_to_merc = existing[(datacenter_id, G)].sort('time_step')[-excess:]
                    for _k_server in servers_to_merc.to_dicts():
                        DC_SERVERS[_k_server["datacenter_id"]] -= 1
                        DC_SLOTS[_k_server["datacenter_id"]] -= _k_server["slots_size"]
                        DC_SCOPED_SERVERS[(_k_server["datacenter_id"], _k_server["server_generation"])] -= 1
                    logger.debug(f"\tExpiring {len(servers_to_merc):5,} of {I}-{G} ({D_ig_old:,} 📉 {D_ig:,})")
                    expiry_list.append(f'{I}-{G}')
                    expiry_pool[G] += [*servers_to_merc['server_id']]

        for I, G, _ in demand_profiles:
            global_servers_in_stock = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) for candidate in DBS[I])
            price_changed = False
            for candidate in DBS[I]:
                datacenter_id = candidate['datacenter_id']
                
                slots_capacity = candidate['slots_capacity']
                base_demand = D_ig = int(IG_base_demand[G].filter(F('time_step') == t)[I].mean() or 0)
                server = get_server_with_selling_price(I, G)
                slots_size = server['slots_size']
                servers_needed_to_meet_demand = D_ig // server['capacity']
                slots_used_in_datacenter = DC_SLOTS.get(datacenter_id, 0)
                assert slots_used_in_datacenter <= slots_capacity, "stock must be <=capacity"

                # servers_in_stock = DC_SCOPED_SERVERS.get((datacenter_id, G), 0)
                # Use global perspective to maximise utilisation (minimise underutilisation)
                servers_in_stock = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) for candidate in DBS[I])
                
                if servers_in_stock < servers_needed_to_meet_demand:
                    
                    capacity_remaining = slots_capacity - slots_used_in_datacenter
                    assert capacity_remaining >= 0, f"capacity remaining ({capacity_remaining}) ought to be >=0"
                    need = np.clip(servers_needed_to_meet_demand - servers_in_stock, 0, capacity_remaining // slots_size)
                    
                    # ASSUME MOVED SERVERS ACT AS FRESH SERVERS (THIS IS FALSE AND MAINTENANCE COST IS MUCH HIGHER, THERE STILL MAY BE NO BREAK EVEN)
                    # Really, we should run some code like `projected_fleet_profit` here instead of blindly harvesting servers that we were going to throw away
                    
                    pool = expiry_pool.get(G, [])
                    amount_to_take_from_pool = min(need, len(pool))
                    existing_capacity = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) * server["capacity"] for candidate in DBS[I])
                    # Assume that when the demand falls below the existing capacity, no capacity will go to the new servers
                    demands = { d['time_step'] : max(0, d[I] - existing_capacity)  for d in IG_base_demand[G]['time_step', I].to_dicts() }
                    
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

                        logger.debug(f"\tHarvesting {len(taken):,} of {I}-{G} from expiry pool for €{len(moved)*server['cost_of_moving']:,} (ẟ{len(taken)*server['capacity']:,} to meet {D_ig:,})")
                            
                        DC_SERVERS[datacenter_id] = DC_SERVERS.get(datacenter_id, 0) + len(taken)
                        DC_SLOTS[datacenter_id] = DC_SLOTS.get(datacenter_id, 0) + len(taken) * slots_size
                        DC_SCOPED_SERVERS[(datacenter_id, G)] = DC_SCOPED_SERVERS.get((datacenter_id, G), 0) + len(taken)

                        actions += move_actions
                        
                    if need > 0:
                        P = server["purchase_price"]
                        
                        # When shit is sunshine and rainbows and there is some stock quantity we can purchase which is profitable even if the minimum demand was sustained for a very long time
                        
                        other_perspective = np.clip(((IG_base_demand[G].filter(F('time_step').is_between(t, t + parameters.expiry_lookahead * 3))[I].min() or 0) - existing_capacity) / server["capacity"] / need, 0.0, 1.0)
                        if other_perspective < 1.0:
                            other_perspectives = [other_perspective]
                            steps_to_consider = 10
                            for i in range(steps_to_consider):
                                if i / steps_to_consider > other_perspective:
                                    other_perspectives.append(i / steps_to_consider)
                        else:
                            other_perspectives = []
                        
                        profitable_scenarios = projected_fleet_profit(
                            n=need,
                            cost_of_energy=candidate["cost_of_energy"],
                            server=server,
                            demands=demands,
                            t=t,
                            break_even_per_server=P * parameters.break_even_coefficient,
                            initial_balance_per_server=-P,
                            ratios=[1.0, *other_perspectives],
                            use_correct_capacity_formulation=True
                        )
                        
                        is_profitable_scenario = len(profitable_scenarios) != 0
                        profitable_scenarios = sorted(profitable_scenarios, key=lambda m: m[1])
                        
                        if is_profitable_scenario and len(profitable_scenarios):
                            best_scale, _ = profitable_scenarios[-1]
                            formatted = [f'{scalar} €{int(profit_earned):,}' for scalar, profit_earned in profitable_scenarios]
                            logger.debug(f"\tBest choice is: '{formatted[-1]}', others: {formatted[:-1]}")
                            need = int(need * best_scale)
                            logger.debug(f"\tBuy {need} more {I}-{G} servers for €{int(need*P):,}!")

                            # pricing_strategy.append(get_default_pricing_strategy_for_demand_segment(I, G, t=t))
                        else:
                            logger.debug(f"\tDon't drop €{int(need*P):,} on {need} more {I}-{G} servers!")
                            
                            if global_servers_in_stock > 0:
                                # There is no point bumping up the price if we have no servers to sell services to customers
                                if base_demand != 0:
                                    
                                    # if base demand is zero (nobody wants these servers anymore), then no amount of selling price
                                    # manipulation will increase profits
                                    demand_met = global_servers_in_stock * server["capacity"]

                                    # breakpoint()
                                    
                                    if demand_met > base_demand:
                                        # In this case we really need to just sell or hold servers until demand goes up again
                                        pass
                                    elif demand_met <= base_demand:
                                        # In this case we can decrease the current demand by increasing the selling prices
                                        demand_delta = demand_met / base_demand - 1
                                        # ratio = demand_met / base_demand - 1
                                        relevant_elasticity = elasticity_IG[I, G]['elasticity'][0]
                                        target_price = server["selling_price"] * (demand_delta / relevant_elasticity + 1)
                                        new_strategy = create_pricing_strategy(I, G, target_price, t)
                                        # new_strategy = get_default_pricing_strategy_for_demand_segment(I, G, t=t)
                                        logger.debug(f"(t={t} {I}-{G}) MET: {demand_met}, BASE: {base_demand}, ΔDᵢg: {demand_delta}, →$: {target_price}, og: {server['selling_price']}")
                                        
                                        if not price_changed:
                                            # PRICING STRATEGY CHANGE
                                            pricing_strategy.append(new_strategy)
                                            
                                        price_changed = True

                            
                        if t >= server["release_start"] and t <= server["release_end"] and is_profitable_scenario:
                            warning = f"([38;2;255;0;0m🤡 {I}-{G} was expired last round! 🤡[m)" if f'{I}-{G}' in old_expiry_list else ""
                            logger.debug(f"\tPurchasing {need:,} of {I}-{G} for €{int(need * server["purchase_price"]):,} (ẟ{need * server["capacity"]:,} to meet {D_ig:,} / {(need+servers_in_stock)*server["capacity"]:,}) {warning}")

                            buy_actions = [
                                {
                                    "time_step" : t,
                                    "datacenter_id" : datacenter_id,
                                    "server_generation" : G,
                                    "server_id" : str(current_server_index := current_server_index + 1),
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

                            delta = parameters.reap_delta
                            dc_servers_to_delete_at[t + server['life_expectancy'] - delta] = dc_servers_to_delete_at.get(t + server['life_expectancy'] - delta, []) + [server['server_id'] for server in buy_actions]
                            
            if not price_changed:
                # PRICING STRATEGY CHANGE
                pricing_strategy.append(get_default_pricing_strategy_for_demand_segment(I, G, t=t))

        excess_ids = []
        for G in server_generations:
            excess_ids += expiry_pool.get(G, [])
        if len(excess_ids) > 0:
            stock = expire_ids(excess_ids, do_book_keeping=False)

        # Figure out how to just query the log level
        if log_info:
            K = pl.concat([demand.filter(F('time_step') == t).select('server_generation', I).rename({ I : 'demand' }).with_columns(latency_sensitivity=pl.lit(I)) for I in latency_sensitivities])
            E = stock.join(servers, on="server_generation") \
                .join(datacenters, on='datacenter_id') \
                .with_columns(energy=F('energy_consumption') * F('cost_of_energy'))['energy'].sum() or 0
            R = stock.join(servers, on="server_generation") \
                .join(selling_prices, on=["latency_sensitivity", "server_generation"]) \
                .join(datacenters, on='datacenter_id') \
                .group_by(["latency_sensitivity", "server_generation"]) \
                .agg(F('capacity').sum(), F('selling_price').mean()) \
                .join(K, on=["server_generation", "latency_sensitivity"]) \
                .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') * F('selling_price'))['capacity'].sum() or 0
            C = stock.join(servers, on="server_generation").filter(F('time_step') == t)['purchase_price'].sum() or 0
            P = (R - C - E) or 0
            U = stock.join(servers, on="server_generation") \
                .select("latency_sensitivity", 'server_generation', 'capacity') \
                .group_by(["latency_sensitivity", "server_generation"]) \
                .agg(F('capacity').sum()) \
                .join(K, on=["server_generation", "latency_sensitivity"]) \
                .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') / F('capacity'))['capacity'].mean() or 0.0
            O = P * U
            logger.debug(f"{t:3}: O: {int(O):<11,}, P: {int(P):<11,} (R={int(R):<11,}, C={int(C):11,},  E={int(E):11,}), U: {U:3.02}")

        if True:
            # bad code above
            #
            # good code below

            augmented_stock = stock.join(servers, on="server_generation") \
                .join(selling_prices, on=["latency_sensitivity", "server_generation"]) \
                .join(datacenters, on=['datacenter_id', 'latency_sensitivity'], how='inner') \
                .group_by(['latency_sensitivity', 'server_generation'])
            
            ig_utilisations = []

            augmented_stock_with_balance = {
                (I, G) : S \
                .join(
                    balance_sheets[-1].select(F('*').exclude('time_step')) if len (balance_sheets) != 0 else pl.DataFrame([], schema={'server_id':pl.String, 'balance':pl.Float64}),
                    on='server_id',
                    how='left'
                ) \
                .with_columns(F('balance').fill_null(0.0)) \
                .with_row_index()
                for [I, G], S in augmented_stock
            }
            
            for (I, G), S in augmented_stock:
                demand_met = len(S) * S['capacity'][0]
                d = IG_base_demand[G].filter(F('time_step') == t)[I]
                d = d.item() if len(d) == 1 else 0.0
                Ud = min(demand_met, d) / demand_met
                ig_utilisations.append(Ud)

                # augment to join with demand
                max_rewarded_services = d // S['capacity'][0]
                
                augmented_stock_with_balance[(I, G)] = augmented_stock_with_balance[(I, G)].with_columns(
                    profit = (pl.when(F('index') < max_rewarded_services) \
                        .then(F('selling_price') * F('capacity')) \
                        .otherwise(0.0)),
                    balance = F('balance')
                    + (pl.when(F('index') < max_rewarded_services) \
                        .then(F('selling_price') * F('capacity')) \
                        .otherwise(0.0)
                    - pl.when(F('time_step') == 1).then(F('purchase_price')).otherwise(0)
                    - F('energy_consumption') * F('cost_of_energy')
                    - pl.struct(['average_maintenance_fee', 'time_step']).map_batches(
                        lambda combined: get_maintenance_cost(
                            combined.struct.field('average_maintenance_fee'),
                            t - combined.struct.field('time_step') + 1,
                            96
                        )
                    ))
                )

            utilisation = kz = sum(ig_utilisations) / len(ig_utilisations) if len(ig_utilisations) > 0 else 1.0

            values = augmented_stock_with_balance.values()
            if augmented_stock_with_balance:
                balance_sheet = pl.concat(augmented_stock_with_balance.values()).select('server_id', 'balance', 'profit').with_columns(time_step= t)
                balance_sheets.append(balance_sheet)
            
        slots = { slot['datacenter_id'] : slot['slots_size'] for slot in stock.group_by('datacenter_id').agg(F('slots_size').sum()).to_dicts() }
        _datacenters = { datacenter['datacenter_id'] : datacenter['slots_capacity'] for datacenter in datacenters.to_dicts() }
        logger.info(f"{t:3}: %s", ", ".join(f"DC{i + 1}: {slots.get(f'DC{i + 1}', 0):6} ({int(100*(slots.get(f'DC{i + 1}', 0)/_datacenters[f'DC{i + 1}'])):3}%)" for i in range(4)))

        if return_stock_log:
            stocks.append('')

        # make equivalents which make caches redundant (please god keep this up to date…)
        if assertions_enabled:
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
            
            for (I, G), chips in stock.group_by(['latency_sensitivity', 'server_generation']):
                logger.debug(f"(t={t} {I}-{G}) db check. count: {len(chips)}, capacity: {len(chips) * get_server_with_selling_price(I, G)['capacity']}")

        old_expiry_list = expiry_list
        
    # actions = k # → 993267430.358398
    # actions = actions # → 993023338.5467423
    
    if return_stock_log:
        return actions, pricing_strategy, stocks
    else:
        return actions, pricing_strategy

if __name__ == "__main__":
    demand, *_ = load_problem_data()
    solution = get_my_solution(demand)
    with open ("kjv.json", "w") as w:
        json.dump(solution, w, indent=1)
