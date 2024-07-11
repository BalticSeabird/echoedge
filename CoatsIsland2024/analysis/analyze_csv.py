import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/Volumes/T7 Shield/Analysis/Hudson Bay 2024/Run 1/combined.csv')
df = df.drop(columns=['Unnamed: 0'])
df['time'] = pd.to_datetime(df['time'])


# Plot results
plt.gca().invert_yaxis()
plt.plot(df['time'], df['depth'])
plt.show()