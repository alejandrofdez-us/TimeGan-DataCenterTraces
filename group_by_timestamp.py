import pandas as pd
import datetime


#start_date = datetime.datetime(2022, 1, 1)


df = pd.read_csv('data/machine_usage_cut_days_3-4-5-6.csv')
#df = df.drop(df.columns[[0,4,5]], axis=1) #drop machine_id and empty columns
df = df.groupby(df.columns[0]).mean()
df = df.drop(df.columns[0], axis=1) # falla!!
df.to_csv('data/machine_usage_grouped_days_3-4-5-6.csv')
#
# for day in range (0,9):
#     print ("Procesando día", day)
#     df_day = df[(df[0] > 8640*day) & (df[0] < 8640*day+1)]
#     df_day = df_day.groupby([0]).mean()
#     df_day.to_csv('data/machine_usage_grouped_day_'+str(day)+'.csv')
#     del df_day

#df['start_hour'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).hour, axis=1)
#df['start_minute'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).minute, axis=1)
#df['start_second'] = df.apply(lambda row: (start_date + datetime.timedelta(seconds=row['time_stamp'])).second, axis=1)

