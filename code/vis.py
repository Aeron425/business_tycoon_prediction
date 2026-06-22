import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

data = pd.read_csv("resources/data/data.csv")

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(data["Time"], data["Money"], color="blue")
ax.set_xlabel("Time")
ax.set_ylabel("Money")
ax.xaxis.set_major_locator(MaxNLocator(nbins=30))
ax.set_title("Over Time Price Changes")

plt.savefig("resources/plots/data.png", dpi=120)
plt.close()