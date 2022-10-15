import pandas as pd

df = pd.read_csv('data/mu_day3.csv')
df = df.drop('machine_id', axis=1)
df.groupby(['time_stamp']).mean().to_csv('data/mu_day3_grouped.csv')