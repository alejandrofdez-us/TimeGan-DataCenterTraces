import csv
import pandas as pd
import datetime

#To start_time for a datetime due to bug: https://github.com/sdv-dev/SDV/issues/943
start_date = datetime.datetime(2022, 1, 1)

raw_dataset = pd.read_csv("data/batch_task.csv")
raw_dataset = raw_dataset[(raw_dataset['end_time'] > 0) & (raw_dataset['status'] == 'Terminated')]

for i in range(0,9):
    print('Procesando día '+str(i))
    filtered_dataset = raw_dataset[(raw_dataset['start_time'] > 86400*i) & (raw_dataset['start_time'] < 86400*(i+1))]
    filtered_dataset['makespan'] = filtered_dataset.apply(lambda row: row['end_time'] - row['start_time'], axis=1)
    trimmed_dataset = filtered_dataset.drop(columns=['task_name', 'status', 'end_time', 'task_type', 'job_name'])
    sorted_dataset = trimmed_dataset.sort_values(by=['start_time'])
    sorted_dataset['start_hour'] = sorted_dataset.apply(lambda row: (start_date + datetime.timedelta(seconds=row['start_time'])).hour, axis=1)
    sorted_dataset['start_minute'] = sorted_dataset.apply(lambda row: (start_date + datetime.timedelta(seconds=row['start_time'])).minute, axis=1)
    final_dataset = sorted_dataset.drop(columns=['start_time'])
    final_dataset.to_csv('data/batch_task_day'+str(i)+'_preprocessed.csv', index=False)



# cabecera: task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem
# | task_name       | string     |       | task name. unique within a job                  |
# | instance_num    | bigint     |       | number of instances                             |
# | job_name        | string     |       | job name                                        |
# | task_type       | string     |       | task type                                       |
# | status          | string     |       | task status                                     |
# | start_time      | bigint     |       | start time of the task                          |
# | end_time        | bigint     |       | end of time the task                            |
# | plan_cpu        | double     |       | number of cpu needed by the task, 100 is 1 core |
# | plan_mem        | double     |       | normalized memorty size, [0, 100]
