import pandas as pd
import datetime


#The start of the trace (which has a timestamp of 600s, not 0s) corresponds to 19:00 on Sunday May 1, 2011,
# in the local timezone of the datacenter.  It happens to be in the US Eastern time zone, so this is EDT.

directory = 'data/trazas_google/'
full_data = pd.read_csv(directory+'instance_usage_full_300secs.csv')
begin_date_offset = (68400-600)*1000000
for day in range (0, 32):
    print("Procesando día", day)
    day_start_time = day*86400*1000000 - begin_date_offset# añadir offset
    day_end_time = day_start_time + 86400*1000000
    aux_day = full_data[(full_data['group_start_time'] >= day_start_time) & (full_data['group_start_time'] < day_end_time)]
    aux_day = aux_day.drop(aux_day.columns[[0]],axis = 1)
    aux_day.to_csv(directory+'instance_usage_5min_sample_day_'+f'{day:02}'+'.csv', index=False)
