import json
import matplotlib.pyplot as plt

with open("trace.json") as f: trace = json.load(f)

plt.style.use('dark_background')
plt.plot(trace)
plt.xlabel("time")
plt.ylabel("profit")
plt.title("Profit over Time")
plt.show()
