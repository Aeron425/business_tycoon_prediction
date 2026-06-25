import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

data = pd.read_csv("resources/data/data.csv")

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(data["time"], data["money"], color="blue")
ax.set_xlabel("time")
ax.set_ylabel("money")
ax.xaxis.set_major_locator(MaxNLocator(nbins=30))
ax.set_title("Over Time Price Changes")

plt.savefig("resources/plots/data.png", dpi=120)
plt.close()

print('Plots saved to resources/plots/')

data["direction"] = data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
series = data["direction"]

max_lag = 100
acf = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]

plt.figure(figsize=(10, 5))
plt.bar(range(1, max_lag + 1), acf)
plt.axhline(0, color='black')
plt.title("Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.savefig("/home/gavin/Desktop/coding_projects/python/business_tycoon_prediction/resources/plots/autocorrelation_with_direction.png")
plt.close()

series = data["money"]

max_lag = 100
acf = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]

plt.figure(figsize=(10, 5))
plt.bar(range(1, max_lag + 1), acf)
plt.axhline(0, color="black")
plt.title("Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.savefig(
    "/home/gavin/Desktop/coding_projects/python/business_tycoon_prediction/resources/plots/autocorrelation_with_money.png"
)
plt.close()