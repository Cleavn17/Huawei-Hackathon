from utils import load_problem_data
import pandas as pd
import numpy as np
import base64, struct
from random import Random
import json

def compress(i):
    return base64.b64encode(struct.pack('<i', i).rstrip(b'\x00')).strip(b'=')

demand, datacenters, servers, selling_prices = load_problem_data()
i = 1
actions = []
server_generations = ["GPU.S1", "GPU.S2", "GPU.S3", "CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4"]
latency_sensitivities = ["low", "high", "medium"]
stock = pd.DataFrame(columns=['server_generation', 'latency_sensitivity', 'datacenter_id', 'server_id', 'time_step', 'slots_size'])

for t in range(168):
    t = t + 1
    merged = pd.merge(stock, servers[["server_generation", "life_expectancy"]], on="server_generation")
    dead_ids = merged[(merged['life_expectancy'] - (t - merged['time_step'])) <= 1]['server_id']
    # can handle demand culling itself (?)
    excess_ids = []
    for server_generation in server_generations:
        for latency_sensitivity in latency_sensitivities:
            d = demand.pivot(index=['time_step', 'latency_sensitivity'], columns=[])[server_generation][t, latency_sensitivity]
            # demand normalised by this server's capacity
            nd = d // servers[servers['server_generation'] == server_generation]['capacity'].item()
            # pick at random if more than one candidate is present
            candidate = datacenters[datacenters['latency_sensitivity'] == latency_sensitivity]
            candidate = candidate.reset_index(drop=True).loc[Random(t).randint(0, len(candidate) - 1)]
            datacenter_id = candidate['datacenter_id']
            slots_capacity = candidate['slots_capacity']
            server = servers[servers['server_generation'] == server_generation].reset_index(drop=True).loc[0]
            slots_size = server['slots_size']
            [release_start, release_end] = json.loads(server['release_time'])
            can_buy = t >= release_start and t <= release_end
            have_df = stock[(stock['datacenter_id'] == datacenter_id) & (stock['server_generation'] == server_generation)]
            have_n = len(have_df)

            have_df_general = stock[stock['datacenter_id'] == datacenter_id]
            have_n_general = have_df_general['slots_size'].sum()
            assert have_n_general <= slots_capacity, "stock must be <=capacity"
            
            capacity_remaining = slots_capacity - have_n_general
            assert capacity_remaining >= 0, f"capacity remaining ({capacity_remaining}) ought to be >=0"
            need = np.clip(nd - have_n, 0, capacity_remaining // slots_size)

            excess = max(0, have_n - nd)
            
            if excess > 1000 or (nd == 0 and excess > 0):
                # only retire the newest servers to keep utilisation metrics up
                excess_ids += [*have_df.iloc[-excess:]['server_id']]
            if excess > 0:
                assert need == 0, "what?"
            if can_buy:
                buy_actions = [
                    {
                        "time_step" : t,
                        "datacenter_id" : datacenter_id,
                        "server_generation" : server_generation,
                        "server_id" : compress(i := i + 1).decode(),
                        "action" : "buy"
                    }
                    for _ in range(need)
                ]
                actions += buy_actions
                stock = pd.concat([stock, pd.DataFrame([{**action, "latency_sensitivity" : latency_sensitivity, "slots_size" : slots_size } for action in buy_actions])])
                stock.reset_index(drop=True, inplace=True)
    
    if len(excess_ids) + len(dead_ids) > 0:
        combination = set([*excess_ids, *dead_ids])
        to_drop = stock[stock['server_id'].isin(combination)]
        for row in to_drop.index:
            server = to_drop.loc[row]
            actions.append({
                "time_step" : t,
                "datacenter_id" : server["datacenter_id"],
                "server_generation" : server["server_generation"],
                "server_id" : server['server_id'],
                "action" : "dismiss"
            })
        stock.drop(to_drop.index, inplace=True)
        stock.reset_index(drop=True, inplace=True)

    slots = stock.groupby(by=['datacenter_id']).agg({'slots_size': 'sum'})['slots_size']
    print(f"{t:3}:", ", ".join(f"DC{i + 1}: {slots.get(f'DC{i + 1}', 0):6}" for i in range(4)))

with open ("kjv.json", "w") as w:
    json.dump(actions, w, indent=2)
