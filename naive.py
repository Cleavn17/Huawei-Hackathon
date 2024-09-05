from utils import load_problem_data
import numpy as np
import base64, struct
import json
import polars as pl
from polars import col as F
from functools import lru_cache
from line_profiler import profile

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
    latency_sensitivities = ["low", "high", "medium"]

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
        IG_existing_sum = { k : v['slots_size'].sum() for k, v in stock.group_by(['latency_sensitivity', 'server_generation']) }
        IG_dmd = { G : v for [G], v in demand.group_by('server_generation') }
        
        for I in latency_sensitivities:
            for G in server_generations:
                # Demand with lookahead into the future
                S = get_server_with_selling_price(I, G)
                D_ig = IG_dmd[G].filter(F('time_step').is_between(t, t + 10))[I].mean() or 0
                C_ig = datacenters.filter(F('latency_sensitivity') == I)['slots_capacity'].sum() - IG_existing_sum.get((I, G), 0)
                A_ig = min(D_ig, C_ig) * S["selling_price"] - S["energy_consumption"] * datacenters.filter(F('latency_sensitivity') == I)['cost_of_energy'].min()
                demand_profiles.append((I, G, A_ig))

        demand_profiles = sorted(demand_profiles, key=lambda p: -p[2])
        
        expiry_pool = {}

        DBS = { k : v.to_dicts() for [k], v in datacenters.group_by('latency_sensitivity') }


        expiry_list = []

        expiry_list_set = set(expiry_list)
        old_expiry_list_set = set(old_expiry_list)
        
        for I, G, _ in demand_profiles:
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
                Z_ig = servers_in_stock * server['capacity']
                try:
                    utilisation = min(D_ig, Z_ig) / Z_ig
                except ZeroDivisionError:
                    utilisation = 1.0
                    
                excess = max(0, servers_in_stock - max(servers_needed_to_meet_demand, servers_needed_to_meet_demand_next))

                if G not in expiry_pool:
                    expiry_pool[G] = []
                
                if excess > 0 and (utilisation < 0.9 or servers_needed_to_meet_demand == 0):
                # if excess > 1000 or (servers_needed_to_meet_demand == 0 and excess > 0):
                    # Don't move unprofitable servers…
                    servers_to_merc = existing[(datacenter_id, G)].sort('time_step')[-excess:]
                    # do book keeping here
                    for server in servers_to_merc.to_dicts():
                        DC_SERVERS[server["datacenter_id"]] -= 1
                        DC_SLOTS[server["datacenter_id"]] -= server["slots_size"]
                        DC_SCOPED_SERVERS[(server["datacenter_id"], server["server_generation"])] -= 1

                    print(f"\tExpiring {len(servers_to_merc):5,} of {I}-{G} ({D_ig_old:,} 📉 {D_ig:,})")

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
                        stock = stock.with_columns(datacenter_id=pl.when(F('server_id').is_in(taken)).then(pl.lit(datacenter_id)).otherwise('datacenter_id')) # just rename all that shit
                        move_actions = []
                        for _server in moved.to_dicts():
                            move_actions.append({
                                "time_step" : t,
                                "datacenter_id" : datacenter_id,
                                "server_generation" : G,
                                "server_id" : _server['server_id'],
                                "action" : "move"
                            })

                        print(f"\tHarvisting {len(taken):,} of {I}-{G} from expiry pool for €{len(moved)*server['cost_of_moving']:,} ({len(taken) - len(moved):,} came free) (ẟ{len(taken)*server['capacity']:,} to meet {D_ig:,})")
                            
                        DC_SERVERS[datacenter_id] = DC_SERVERS.get(datacenter_id, 0) + len(taken)
                        DC_SLOTS[datacenter_id] = DC_SLOTS.get(datacenter_id, 0) + len(taken) * slots_size
                        DC_SCOPED_SERVERS[(datacenter_id, G)] = DC_SCOPED_SERVERS.get((datacenter_id, G), 0) + len(taken)

                        actions += move_actions
                    
                    # Do not purchase chips which will never make
                    # money to increase endgame P and normalised L
                    # (due to a lack of new chips).
                    C = server["capacity"]
                    P = server["purchase_price"]
                    R = server["selling_price"]
                    T = min(60, 168 - t)
                    energy_costs = T * server["energy_consumption"] * candidate["cost_of_energy"]
                    maintenance_costs = sum(get_maintenance_cost(server["average_maintenance_fee"], i + 1, server["life_expectancy"]) for i in range(T))
                    purchasing_price = P
                    maximum_feasible_profit = C * R * T
                    # The only guaranteed cost is purchasing
                    # price. Profit might be rewarded if longevity and
                    # utilisation increase .etc. Just using a random
                    # coefficient here to dampen their influence
                    # slightly.
                    B = maximum_feasible_profit - purchasing_price - (energy_costs - maintenance_costs) * 0.8
                    chip_is_possibly_profitable = B > 0

                    if t >= server["release_start"] and t <= server["release_end"] and chip_is_possibly_profitable and need > 0:
                        if f'{I}-{G}' in old_expiry_list:
                            warning = f"([38;2;255;0;0m🤡 {I}-{G} was expired last round! 🤡[m)"
                        else:
                            warning = ""
                        print(f"\tPurchasing {need:,} of {I}-{G} for €{int(need * server["purchase_price"]):,} (ẟ{need * server["capacity"]:,} to meet {D_ig:,}) {warning}")
                        
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
        print(f"{t:3}: O: {int(O):<11,}, P: {int(P):<11,} (R={int(R):<11,}, C={int(C):11,},  E={int(E):11,}), L: {L:3.02}, U: {U:3.02}")

        slots = { slot['datacenter_id'] : slot['slots_size'] for slot in stock.group_by('datacenter_id').agg(F('slots_size').sum()).to_dicts() }
        _datacenters = { datacenter['datacenter_id'] : datacenter['slots_capacity'] for datacenter in datacenters.to_dicts() }
        print(f"{t:3}:", ", ".join(f"DC{i + 1}: {slots.get(f'DC{i + 1}', 0):6} ({int(100*(slots.get(f'DC{i + 1}', 0)/_datacenters[f'DC{i + 1}'])):3}%)" for i in range(4)))

        # make equivalents which make caches redundant (please god keep this up to date…)
        db_scoped_servers = { k : len(v) for k, v in stock.group_by(['datacenter_id', 'server_generation']) }
        db_servers = { k : len(v) for [k], v in stock.group_by('datacenter_id') }
        db_slots = { k : v['slots_size'].sum() for [k], v in stock.group_by('datacenter_id') }
        for id in _datacenters.keys():
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
