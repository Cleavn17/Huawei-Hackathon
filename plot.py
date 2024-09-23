import json
from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt

plt.style.use('dark_background')

seed = 1097

def plot_single_stream_of_profit():
    with open("etrace.json") as f: trace = json.load(f)
    figure, axis = plt.subplots()
    axis.plot(trace)
    axis.set_xlabel("time")
    axis.set_ylabel("profit")
    axis.set_title("Profit over Time")
    axis.legend(['Profit'])
    figure.savefig("plot.png")
    plt.show()

def plot_convergence_trace():
    I, G = 'low', 'CPU.S1'
    key = f"{I}-{G}"
    
    with open(f"convergetrace-{seed}.json") as f:
        trace = json.load(f)
        
    demands = [row[key]["demand_"] if key in row else 0.0 for row in trace]
    original_demand = [row[key]["original_demand"] if key in row else 0.0 for row in trace]
    selling_prices = [row[key]["price"] * 50_000 if key in row else 0.0 for row in trace]
    capacities = [row[key]["capacity"] if key in row else 0.0 for row in trace]
        
    figure, axis = plt.subplots()

    axis.plot(demands)
    axis.plot(original_demand, linestyle=":")
    axis.plot(selling_prices)
    axis.plot(capacities)
    
    axis.set_xlabel("Time")
    axis.set_ylabel("Demand Units")
    axis.set_title(f"Demand Control Characteristics over Time ({G} '{I}')")
    axis.legend(['Demand', 'Original Demand', 'Adjusted Price', 'Capacity'])
    figure.savefig(f"plot-convergence-{seed}.png")
    plt.show()

if __name__ == "__main__":
    # plot_single_stream_of_profit()
    plot_convergence_trace()
