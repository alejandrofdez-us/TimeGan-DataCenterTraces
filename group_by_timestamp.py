import pandas as pd
import datetime


start_date = datetime.datetime(2022, 1, 1)


df = pd.read_csv('data/mu_day3.csv')
df = df.drop('machine_id', axis=1)
df['start_hour'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).hour, axis=1)
df['start_minute'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).minute, axis=1)
df['start_second'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).second, axis=1)
df = df.groupby(['time_stamp']).mean()

df.to_csv('data/mu_day3_grouped_hours.csv')
