import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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