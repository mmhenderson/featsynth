import numpy as np
from datetime import datetime, timedelta

def adjust_datetime_str(date_str, hours, time_format_str = '%d/%m/%Y %H:%M:%S'):
    date_num = datetime.strptime(date_str, time_format_str)
    date_num_adj = date_num + timedelta(hours=-8)
    date_str_adj = datetime.strftime(date_num_adj,  time_format_str)
    return date_str_adj

def argsort_dates(date_str_list, time_format_str = '%d/%m/%Y %H:%M:%S'):
    date_num = [datetime.strptime(date_str, time_format_str) for date_str in date_str_list]
    return np.argsort(date_num)
