# caith do dhochas agus do ghui sa bhruascar
# nil aon usaid acu
# is ifreann te gan reasun
# an cod faoi bhun

import json
import itertools
import math
from utils import load_problem_data
import numpy as np
import base64, struct
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

rates = [0.0665550253495761, 0.07504024917250507, 0.07373038829090167, 0.05190032306591112, 0.05107575970000427, 0.06059422512949742, 0.08202931565454682, 0.0821765794338332, 0.05601625081065526, 0.067432230682308, 0.07919241177651139, 0.07086069501808555, 0.06044047462336472, 0.052946380038025036, 0.06631051788617792, 0.0651922869472055, 0.05810499777431877, 0.06448663412859632, 0.08253235907162056, 0.05703814452823927, 0.05316508357212482, 0.0910878801203621, 0.06299402134302048, 0.08596277330404743, 0.05054682445964736, 0.06659473933986396, 0.05879498478654589, 0.051905732565933324, 0.08999666524767898, 0.08225586339008775, 0.06664182858010569, 0.08415429232759447, 0.07050353450959125, 0.06861744838547738, 0.060845034296460904, 0.05794649045996698, 0.062074363925357146, 0.05787656403362245, 0.054201070221407605, 0.058045335673534, 0.0647980705203653, 0.06608136433470334, 0.06649068116313343, 0.053641666652467404, 0.08714379662323299, 0.06780852647284875, 0.05810120283502803, 0.07608599507232297, 0.054328206310945, 0.05282238579004179, 0.05604596442754132, 0.07824503957338812, 0.057248575908624685, 0.06261025700574407, 0.07984493679711775, 0.07183822267048479, 0.07115474929905123, 0.07941522255297147, 0.09240693852536694, 0.05849409926526266, 0.08291495580938972, 0.05327770477443257, 0.09972301393402984, 0.08061975135948017, 0.06812100111318882, 0.050230656920518986, 0.0691738784254954, 0.072060581005856, 0.07195276053167608, 0.06049658033797831, 0.05156577242391103, 0.08478276848324609, 0.09188737221381407, 0.061066038027204256, 0.07077983271333399, 0.05707080456729906, 0.0503203510857775, 0.08421748225388569, 0.09784521630817809, 0.05854711222556484, 0.058479216239125414, 0.08140864185278462, 0.09817907531266197, 0.05974663755142876, 0.05059156834399523, 0.09739415718014305, 0.08249853407487008, 0.06912800569318002, 0.05015431678470312, 0.07386577421008589, 0.09832467701341252, 0.09508174888691762, 0.07972504610137235, 0.0824545777139264, 0.09395859017362237, 0.070630819563012, 0.0798590137930202, 0.08587960779165359, 0.0853559873482121, 0.0698323880400669, 0.08576068177926524, 0.05203248292778784, 0.058954980843170926, 0.05629285180139859, 0.07026287334450444, 0.09412945245484151, 0.05554464909721682, 0.06205892326560348, 0.07271534476720921, 0.07092764847444319, 0.07551145402638079, 0.09477652254380205, 0.05746882949857669, 0.07954515871968779, 0.08060847501891383, 0.07605433647883032, 0.06909440300301785, 0.06883561837681934, 0.09459488881475149, 0.06251195761763888, 0.07247256505504246, 0.05810842503060285, 0.07295804445264449, 0.05331499930843036, 0.05364985869329036, 0.05099583198091994, 0.0692646571924726, 0.09293533125974725, 0.08042037017221931, 0.05264528802656404, 0.08808959077134378, 0.08193701865504383, 0.05239701112012681, 0.07419741216795599, 0.05633832512399516, 0.08980110494882257, 0.07971387006871095, 0.07801381217108501, 0.06694745002259911, 0.054436140142313134, 0.09374581273818475, 0.06693579978719905, 0.0723204876344245, 0.05775127108252804, 0.059154741171617696, 0.06634801397887449, 0.06365761510983277, 0.082889317867915, 0.0907893281622972, 0.06678720923859091, 0.06312208629196121, 0.07238304677566258, 0.08175841328458903, 0.09452315286182027, 0.05111199183340636, 0.05560600039271329, 0.05513216202166291, 0.09152011390551341, 0.09650091628157766, 0.06888189957341317, 0.0537875680687065, 0.07032447038300865, 0.05864760186798964, 0.08870830640940318, 0.07142827078147859, 0.08847508677397034, 0.055477353461670105, 0.09535631863465348, 0.0670029979306786, 0.08021931557876276, 0.08520591314416764, 0.054489320325494535, 0.0924312125799636, 0.05255528337989817, 0.0687865845003437, 0.057771569624329756, 0.06490285162007996, 0.09917652871396153, 0.052441150260447496, 0.0639397758694379, 0.07698978503257425, 0.06242900672857499, 0.08165515742421563, 0.06297813767576374, 0.052563977907039636, 0.09104203120251311, 0.06557995375128141, 0.06082887680348569, 0.09492642686598678, 0.06452211559438269, 0.07079225775219387, 0.06480471579469109, 0.059521639467189864, 0.05200584960064997, 0.05552070059363882, 0.085569960093765, 0.09993176908993795, 0.08050596989583203, 0.08214250525655287, 0.05646706720068601]

@profile
def get_my_solution(
        demand,
        assertions_enabled = False,
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

        modified_prices = {}
        modified_demands = {}
        
        for I, G, _ in demand_profiles:
            global_servers_in_stock = sum(DC_SCOPED_SERVERS.get((candidate['datacenter_id'], G), 0) for candidate in DBS[I])
            new_strategy = None

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
                                        
                                        logger.debug(f"(t={t} {I}-{G}) MET: {demand_met}, BASE: {base_demand}, ΔDᵢg: {demand_delta}, →$: {target_price}, og: {server['selling_price']}")
                                        if new_strategy is None:
                                            new_strategy = create_pricing_strategy(I, G, target_price, t)
                                            # new_strategy = get_default_pricing_strategy_for_demand_segment(I, G, t=t)
                                            pass
                                        
                                        modified_prices[(I, G)] = float(target_price)
                                        modified_demands[(I, G)] = int(demand_met)
                            
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

            pricing_strategy.append(get_default_pricing_strategy_for_demand_segment(I, G, t=t) if new_strategy is None else new_strategy)

        excess_ids = []
        for G in server_generations:
            excess_ids += expiry_pool.get(G, [])

        if len(excess_ids) > 0:
            stock = expire_ids(excess_ids, do_book_keeping=False)

        # Figure out how to just query the log level
        if log_info:
            restrucutred_demand = pl.concat(
                [
                    demand \
                    .filter(F('time_step') == t) \
                    .select('server_generation', I) \
                    .rename({ I : 'demand' }) \
                    .with_columns(latency_sensitivity=pl.lit(I))
                    for I in latency_sensitivities
                 ]
            )

            modified_restructured_demand = []
            for [G], k in demand.filter(F('time_step') == t).group_by('server_generation'):
                for I in latency_sensitivities:
                    modified_restructured_demand.append({
                        'server_generation': G,
                        'demand' : modified_demands.get((I, G), k[I].item()),
                        'latency_sensitivity' : I,
                    })
            modified_restructured_demand = pl.DataFrame(modified_restructured_demand)
            restrucutred_demand = modified_restructured_demand
            
            E = stock.join(servers, on="server_generation") \
                .join(datacenters, on='datacenter_id') \
                .with_columns(energy=F('energy_consumption') * F('cost_of_energy'))['energy'].sum() or 0

            modified_selling_prices = []
            for (I, G), v in selling_prices.group_by('latency_sensitivity', 'server_generation'):
                modified_selling_prices.append({
                    'latency_sensitivity' : I,
                    'server_generation' : G,
                    'selling_price' : modified_prices.get((I, G), v['selling_price'][0]),
                })
            modified_selling_prices_df = pl.DataFrame(modified_selling_prices)
            
            R = stock.join(servers, on="server_generation") \
                .join(modified_selling_prices_df, on=["latency_sensitivity", "server_generation"]) \
                .join(datacenters, on='datacenter_id') \
                .group_by(["latency_sensitivity", "server_generation"]) \
                .agg(F('capacity').sum(), F('selling_price').mean()) \
                .join(restrucutred_demand, on=["server_generation", "latency_sensitivity"]) \
                .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') * F('selling_price'))['capacity'].sum() or 0
            C = stock.join(servers, on="server_generation").filter(F('time_step') == t)['purchase_price'].sum() or 0
            P = (R - C - E) or 0
            U = stock.join(servers, on="server_generation") \
                .select("latency_sensitivity", 'server_generation', 'capacity') \
                .group_by(["latency_sensitivity", "server_generation"]) \
                .agg(F('capacity').sum()) \
                .join(restrucutred_demand, on=["server_generation", "latency_sensitivity"]) \
                .select("latency_sensitivity", 'server_generation', pl.min_horizontal('capacity', 'demand') / F('capacity'))['capacity'].mean() or 0.0
            O = P
            logger.debug(f"{t:3}: O: {int(O):<11,}, P: {int(P):<11,} (R={int(R):<11,}, C={int(C):11,},  E={int(E):11,}), U: {U:3.02}")
            trace.append(P)

        if True:
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

    with open("trace.json", "w") as f:
        json.dump(trace, f)
        
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
