import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import datetime

data = pd.read_csv("resources/data/model_ready_data.csv")
data.columns = data.columns.str.strip()
data['time'] = pd.to_datetime(data['time'])

plt.figure(figsize = (12, 4))
autocorrelation_plot(data['money'])
plt.title('Autocorrelation')
plt.tight_layout()
plt.savefig('resources/plots/autocorrelation.png', dpi = 120)
plt.close()

plt.figure(figsize = (8, 4))
data['money'].hist(bins = 20, color = 'steelblue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('resources/plots/price_distribution.png', dpi = 120)
plt.close()


plt.figure(figsize = (10, 4))
data.groupby(data['time'].dt.hour)['money'].mean().plot(kind = 'bar', color = 'steelblue')
plt.title('Average Price by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.tight_layout()
plt.savefig('resources/plots/price_by_hour.png', dpi = 120)
plt.close()


plt.figure(figsize = (10, 4))
data.groupby('money')['direction'].mean().plot(kind = 'bar', color = 'steelblue')
plt.title('Average Direction by Price Level')
plt.xlabel('Price')
plt.ylabel('Average Direction')
plt.tight_layout()
plt.savefig('resources/plots/direction_by_price.png', dpi = 120)
plt.close()

print('Plots saved to resources/plots/')