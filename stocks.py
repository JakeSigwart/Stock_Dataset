import os
import time
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from Stock_dataset import *

path = os.path.dirname(__file__)
todays_date = str(dt.date.today())

tickers = ['AAPL', 'AMZN', 'NVDA', 'GM', 'T', 'CAH']
#sp_500_tickers = np.load(path + '\\data\\tickers.npy')
dataset = Stock_dataset(tickers, path+'\\data\\data.pkl', path+'\\data\\dates.pkl', path+'\\data\\proc.npy')

#dataset.quandl_api_key("YOUR API KEY HERE")

data, dates = dataset.fetch_data('2017-01-01', '2017-07-01')
dataset.save_new_data(data, dates, True)
numeric_data, _ = dataset.organize_data_from_vars(data, dates)
proc_data, processed_data_stock, processed_data_dates, combined_dates = dataset.process_data(numeric_data, dates, False)
#processed_data, dates = dataset.update_data(todays_date)  #Un-comment this and comment the above 4 lines after processing first data fetch

num_dates = len(dates)
print(tickers)
print('Data metrics for date: ' + str(dates[num_dates-1]))
print(proc_data[num_dates-1])


