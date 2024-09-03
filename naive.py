from utils import load_problem_data
import numpy as np
import base64, struct
import json
import polars as pl
from polars import col as F
from functools import lru_cache

def compress(i):
    return base64.b64encode(struct.pack('<i', i).rstrip(b'\x00')).strip(b'=')

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
    
    for t in range(168):
        t = t + 1
        existing = { key : data for key, data in stock.group_by(['datacenter_id', 'server_generation']) }
        excess_ids = []
        demand_profiles = []

        # Increases profits by about 10 million
        for I in latency_sensitivities:
            for G in server_generations:
                # Demand with lookahead into the future
                D_ig = demand.filter(F('time_step').is_between(t, t + 10) & (F('latency_sensitivity') == I))[G].mean()
                C_ig = datacenters.filter(F('latency_sensitivity') == I)['slots_capacity'].sum() - stock.filter((F('latency_sensitivity') == I) & (F('server_generation') == G))['slots_size'].sum()
                A_ig = min(D_ig, C_ig) * get_server_with_selling_price(I, G)["selling_price"]
                demand_profiles.append((I, G, A_ig))

        demand_profiles = sorted(demand_profiles, key=lambda p: -p[2])

        for I, G, _ in demand_profiles:

            # pick at random if more than one candidate is present
            candidates = datacenters.filter(F('latency_sensitivity') == I).to_dicts()

            for candidate in candidates:
                datacenter_id = candidate['datacenter_id']
                slots_capacity = candidate['slots_capacity']

                dmnd = int(demand.filter(F('time_step').is_between(t, t + 10) & (F('latency_sensitivity') == I))[G].mean())

                server = get_server_with_selling_price(I, G)
                slots_size = server['slots_size']

                # demand normalised by this server's capacity
                servers_needed_to_meet_demand = dmnd // server['capacity'] // len(candidates)

                # servers_in_datacenter = stock.filter((F('datacenter_id') == datacenter_id))
                servers_in_datacenter = DC_SERVERS.get(datacenter_id, 0)
                # slots_used_in_datacenter = servers_in_datacenter['slots_size'].sum()
                slots_used_in_datacenter = DC_SLOTS.get(datacenter_id, 0)
                assert slots_used_in_datacenter <= slots_capacity, "stock must be <=capacity"

                # scoped to the current generation G
                # servers_in_stock = servers_in_datacenter.filter((F('server_generation') == G))
                servers_in_stock = DC_SCOPED_SERVERS.get((datacenter_id, G), 0)

                excess = max(0, servers_in_stock - servers_needed_to_meet_demand)

                if excess <= 0:
                    capacity_remaining = slots_capacity - slots_used_in_datacenter
                    assert capacity_remaining >= 0, f"capacity remaining ({capacity_remaining}) ought to be >=0"
                    need = np.clip(servers_needed_to_meet_demand - servers_in_stock, 0, capacity_remaining // slots_size)

                    if t >= server["release_start"] and t <= server["release_end"]:
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

                        DC_SERVERS[datacenter_id] = servers_in_datacenter + len(bought)
                        DC_SLOTS[datacenter_id] = slots_used_in_datacenter + len(bought) * slots_size
                        DC_SCOPED_SERVERS[(datacenter_id, G)] = servers_in_stock + len(bought)
                        
                        delta = 5
                        DC_DEAD_AT[t + server['life_expectancy'] - delta] = DC_DEAD_AT.get(t + server['life_expectancy'] - delta, []) + [server['server_id'] for server in buy_actions]

                elif excess > 1000 or (servers_needed_to_meet_demand == 0 and excess > 0):
                    # only retire the newest servers to keep utilisation metrics up
                    excess_ids += [*existing[(datacenter_id, G)].sort('time_step')[-excess:]['server_id']]


        dead_ids = DC_DEAD_AT.get(t, [])

        if len(excess_ids) + len(dead_ids) > 0:
            combination = set([*excess_ids, *dead_ids])
            drop_condition = F('server_id').is_in(combination)
            for server in stock.filter(drop_condition).to_dicts():
                actions.append({
                    "time_step" : t,
                    "datacenter_id" : server["datacenter_id"],
                    "server_generation" : server["server_generation"],
                    "server_id" : server['server_id'],
                    "action" : "dismiss"
                })

                DC_SERVERS[server["datacenter_id"]] -= 1
                DC_SLOTS[server["datacenter_id"]] -= server["slots_size"]
                DC_SCOPED_SERVERS[(server["datacenter_id"], server["server_generation"])] -= 1

            stock = stock.filter(~drop_condition)

        slots = { slot['datacenter_id'] : slot['slots_size'] for slot in stock.group_by('datacenter_id').agg(F('slots_size').sum()).to_dicts() }
        print(f"{t:3}:", ", ".join(f"DC{i + 1}: {slots.get(f'DC{i + 1}', 0):6}" for i in range(4)))

    return actions

if __name__ == "__main__":
    demand, *_ = load_problem_data()
    solution = get_my_solution(demand)
    with open ("kjv.json", "w") as w:
        json.dump(solution, w, indent=1)
