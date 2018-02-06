import os
import time
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import quandl
import pandas_datareader.data as web

#Class to enable neat manipulation and processing of stock data-set. Keep processed data up-to-date.
#Using Quandl.com API to fetch data from the WIKI dataset
#Warning: Save data for earlier dates first then later ones (Use Update method when possible)
class Stock_dataset:
	#Sets up file paths
	def __init__(self, tickers, data_path, dates_path, processed_data_path):
		self.tickers = tickers
		self.num_tickers = len(tickers)
		self.data_path = data_path
		self.dates_path = dates_path
		self.processed_data_path = processed_data_path
		self.WIKI_column_headings = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'ex-dividend', 'split_ratio', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']
		self.headings = ['ticker', 'date', 'ex-dividend', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']
	
	#Set the quandl API Key.
	#Note: Some data is available without a quandl account. These functions require a free quandl account with a valid API Key
	def quandl_api_key(self, key):
		quandl.ApiConfig.api_key = key
	
	#Purpose: Fetch up-to-date stock data using the Quandl.com API
	def fetch_data(self, start_date, end_date):
		num_tickers = len(self.tickers)
		new_dates = []
		new_data = pd.DataFrame(columns=self.headings)
		#Loop through the stocks and get data
		print('Fetching new data in range: ' + start_date + '  ' + end_date)
		for i in range(0, num_tickers):
			print('Retreiving Data For: ' + self.tickers[i])
			temp_data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': self.headings }, ticker = self.tickers[i], date = { 'gte': start_date, 'lte': end_date })
			new_data = new_data.append(temp_data, ignore_index=True)
		#Extract the dates
		unique_new_dates = pd.unique(new_data['date'].copy())
		num_new_dates = len(unique_new_dates)
		
		#Loop through dates. Convert dates to strings (yyyy-mm-dd)
		for n in range(0, num_new_dates):
			date = pd.Timestamp(unique_new_dates[n])
			string_date = str(date)
			string_date = string_date[0:10]
			new_dates.append(string_date)
		#Save data and dates
		print('Retreived New Data Via Quandl API.')
		return new_data, new_dates
		
	#Purpose: Save recently fetched data and dates
	#Input: pandas data-frame (with stock data), dates, bool (True: append new data to old data)
	def save_new_data(self, new_data, new_dates, add_to_existing):
		if add_to_existing and os.path.isfile(self.data_path) and os.path.isfile(self.dates_path):
			with open(self.data_path, mode='rb') as file:
				old_data = pickle.load(file, encoding='bytes')
			with open(self.dates_path, mode='rb') as file:
				dates = pickle.load(file, encoding='bytes')
			data = old_data.append(new_data, ignore_index=True)
			dates.extend(new_dates)
			output = open(self.data_path, 'wb')
			pickle.dump(data, output)
			output = open(self.dates_path, 'wb')
			pickle.dump(dates, output)
		else:
			output = open(self.data_path, 'wb')
			pickle.dump(new_data, output)
			output = open(self.dates_path, 'wb')
			pickle.dump(new_dates, output)
		
	#Purpose: Open data and dates from files
	def load_data(self):
		if os.path.isfile(self.data_path):
			with open(self.data_path, mode='rb') as file:
				data = pickle.load(file, encoding='bytes')
			if os.path.isfile(self.dates_path):
				with open(self.dates_path, mode='rb') as file:
					dates = pickle.load(file, encoding='bytes')
			else:
				print('No dates file found. Creating file from data...')
				dates_df = pd.unique(data['date'].copy())
				num_new_dates = len(dates_df)
				dates = []
				for n in range(0, num_new_dates):
					date = pd.Timestamp(dates_df[n])
					string_date = str(date)
					string_date = string_date[0:10]
					dates.append(string_date)
				output = open(self.dates_path, 'wb')
				pickle.dump(dates, output)
		return data, dates
		
	#Purpose: To sort data by date (Note: Sorting data is the most time consuming step)
	#Input: from_file_=True: load data from file. False: return data already in class
	#Output: Returns the numpy array of numeric data. Shape: [date, stock, num_metrics=6],  Dates array
	def organize_data_from_file(self):
		if os.path.isfile(self.data_path):
			with open(self.data_path, mode='rb') as file:
				data = pickle.load(file, encoding='bytes')
			if os.path.isfile(self.dates_path):
				with open(self.dates_path, mode='rb') as file:
					dates = pickle.load(file, encoding='bytes')
			else:
				print('No dates file found. Creating file now...')
				dates_df = pd.unique(data['date'].copy())
				num_new_dates = len(dates_df)
				dates = []
				for n in range(0, num_new_dates):
					date = pd.Timestamp(dates_df[n])
					string_date = str(date)
					string_date = string_date[0:10]
					dates.append(string_date)
				output = open(self.dates_path, 'wb')
				pickle.dump(dates, output)
		else:	
			print('Error: No data file found!')
		
		num_dates = len(dates)
		numeric_data = np.zeros(shape=[num_dates, self.num_tickers, 6], dtype=float)
		
		for cdate in range(0, num_dates):
			date = pd.Timestamp(dates[cdate])
			print('Loading for: ' + str(date))
			data_for_date = data.loc[data['date']== date].copy()
			for cstock in range(0, self.num_tickers):
				sym = self.tickers[cstock]
				data_slice = data_for_date.loc[data_for_date['ticker'] == sym].copy()
				data_slice = data_slice.drop(columns=['ticker', 'date'])
				if len(data_slice)!=0:
					numeric_data[cdate,cstock,:] = data_slice.as_matrix(columns=['ex-dividend', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'])
				else:
					numeric_data[cdate,cstock,:] = np.zeros(shape=[6], dtype=float)
				data_for_date.drop[data_for_date.ticker == sym] #Abandon rows for the symbol and date to speed up searches
			data.drop[data.date == date]   #Abandon rows for that date to shorten searches
		print('Finished Organizing data loaded from file')
		return numeric_data, dates
	
	#Purpose: To sort data by date (Note: Sorting data is the most time consuming step)
	#Input: pandas data-frame (containing WIKI stock data), dates array (List of unique dates with data)
	#Output: Returns the numpy array of numeric data. Shape: [date, stock, num_metrics=6],  Dates array
	def organize_data_from_vars(self, data, dates):
		num_dates = len(dates)
		numeric_data = np.zeros(shape=[num_dates, self.num_tickers, 6], dtype=float)
		for cdate in range(0, num_dates):
			date = pd.Timestamp(convert_string_to_datetime(dates[cdate]))
			print('Loading for: ' + str(date))
			data_for_date = data.loc[data['date']== date].copy()
			for cstock in range(0, self.num_tickers):
				sym = self.tickers[cstock]
				data_slice = data_for_date.loc[data_for_date['ticker'] == sym].copy()
				data_slice = data_slice.drop(columns=['ticker', 'date'])
				if len(data_slice)!=0:
					numeric_data[cdate,cstock,:] = data_slice.as_matrix(columns=['ex-dividend', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'])
				else:
					numeric_data[cdate,cstock,:] = np.zeros(shape=[6], dtype=float)
				#data_for_date.drop[data_for_date.ticker == sym] #Abandon rows for the symbol to speed up searches
			#data.drop[data.date == date] #Abandon rows for that date to shorten searches
		print('Finished Organizing Data from Variables')
		return numeric_data, dates
	
	#Purpose: Calculate data metrics. These metrics have averages which should be approximately constant through all time 
	#Input: new: organized numeric data, new dates, bool (indicates intent to append to existing dataset)
	#Output: processed_data: shape=[dates,stocks,10], Columns: [percent_gl, open_open_gl, mid_day_gl, intraday_gl, div_yield, HML, volume_ratio, value_percent, percent_over_market, day_beta]
	#processed_data_dates shape=[dates,6], Columns: [total_trading_volume, total_value, percent_average, percent_standard_dev, Index_of_greatest_gain, Index_of_worst_loss]
	#processed_data_stock: shape=[stocks, 2], Columns: [all time avg daily g/l, all time beta]
	#Returns the combined processed dataset, stock_data, date_data, dates
	def process_data(self, numeric_data, dates, add_to_existing):
		print('Processing data. Calculating metrics for stocks...')
		processed_data_exists = os.path.isfile(self.processed_data_path) #There is already processed_data saved
		#Get data from the last date in order to calculate metrics for the first new date (previous close is required to calc %g/l)
		if add_to_existing and processed_data_exists:
			with open(self.data_path, mode='rb') as file:
				old_data = pickle.load(file, encoding='bytes')
			with open(self.dates_path, mode='rb') as file:	
				old_dates = pickle.load(file, encoding='bytes')
			num_old_dates = len(old_dates)
			#Day -1 data: get data for the last date in the old dataset
			last_date = pd.Timestamp(old_dates[num_old_dates-1])
			data_for_last_date = old_data.loc[old_data['date']== last_date].copy()
			last_date_data = np.zeros(shape=[self.num_tickers, 6], dtype=float)
			for cstock in range(0, self.num_tickers):
				sym = self.tickers[cstock]
				data_slice = data_for_last_date.loc[data_for_last_date['ticker'] == sym].copy()
				data_slice = data_slice.drop(columns=['ticker', 'date'])
				if len(data_slice)!=0:
					last_date_data[cstock,:] = data_slice.as_matrix(columns=['ex-dividend', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'])
				else:
					last_date_data[cstock,:] = np.zeros(shape=[6], dtype=float)
		num_dates = len(dates)
		processed_data = np.zeros(shape=[num_dates, self.num_tickers, 10], dtype=float)
		processed_data_dates = np.zeros(shape=[num_dates, 6])
		processed_data_stock = np.zeros(shape=[self.num_tickers, 2])
		
		for day in range(0, num_dates):
			processed_data_dates[day,0] = np.sum(numeric_data[day,:,5]) #Total Traded volume
			total_val = 0
			for stock_c in range(0, self.num_tickers):
				total_val = total_val + numeric_data[day,stock_c,1]*numeric_data[day,stock_c,1]
			processed_data_dates[day,1] = total_val  #Total value of all trades combined
			total_val = 0
		
		#Day 0: Use old data to find data for first date. Do only for stocks with non-zero price
		if add_to_existing and processed_data_exists:
			for stock_c in range(0, self.num_tickers):
				if last_date_data[stock_c,1]!=0:
					yesterday_open = last_date_data[stock_c,1]
					yesterday_close = last_date_data[stock_c,4]
					today_open = numeric_data[0,stock_c,1]
					today_close = numeric_data[0,stock_c,4]
					high = numeric_data[0,stock_c,2]
					low = numeric_data[0,stock_c,3]
					tot_volume = processed_data_dates[0,0]
					tot_value = processed_data_dates[0,1]
					
					percent_gl = (today_close - yesterday_close) / yesterday_close #Standard stock percentage gain/loss
					open_open_gl = (today_open - yesterday_open) / yesterday_open
					mid_day_gl = (today_close - today_open) / today_open
					intraday_gl = (today_open - yesterday_close) / yesterday_close
					div_yield = numeric_data[0,stock_c,0] / yesterday_close
					HML = (high - low) / today_open #High minus low (normalized to day open)
					volume_ratio = numeric_data[0,stock_c,5] / tot_volume #Stock Trading volume divided by volume of all stocks for day
					value_percent = today_open*numeric_data[0,stock_c,5] / tot_value  #Value of stock traded today / Total value traded for all sp500 stocks 
					processed_data[0,stock_c,0:8] = [percent_gl, open_open_gl, mid_day_gl, intraday_gl, div_yield, HML, volume_ratio, value_percent]
		
		for stock_c in range(0, self.num_tickers):
			print('First pass. Stock: ' + self.tickers[stock_c])
			for day in range(1, num_dates):
				#Calculate only if stock price was non-zero
				if numeric_data[day-1,stock_c,1] > 0:
					yesterday_open = numeric_data[day-1,stock_c,1]
					yesterday_close = numeric_data[day-1,stock_c,4]
					today_open = numeric_data[day,stock_c,1]
					today_close = numeric_data[day,stock_c,4]
					high = numeric_data[day,stock_c,2]
					low = numeric_data[day,stock_c,3]
					tot_volume = processed_data_dates[day,0]
					tot_value = processed_data_dates[day,1]
					
					percent_gl = (today_close - yesterday_close) / yesterday_close #Standard stock percentage gain/loss
					open_open_gl = (today_open - yesterday_open) / yesterday_open
					mid_day_gl = (today_close - today_open) / today_open
					intraday_gl = (today_open - yesterday_close) / yesterday_close
					div_yield = numeric_data[day,stock_c,0] / yesterday_close
					HML = (high - low) / today_open #High minus low (normalized to day open)
					volume_ratio = numeric_data[day,stock_c,5] / tot_volume #Trading volume divided by volume of all stocks for day
					value_percent = today_open*numeric_data[day,stock_c,5] / tot_value #Fraction of total sp500 value, un-weighted by volume
					processed_data[day,stock_c,0:8] = [percent_gl, open_open_gl, mid_day_gl, intraday_gl, div_yield, HML, volume_ratio, value_percent]
			
		for day in range(0, num_dates):
			processed_data_dates[day,2] = np.average(processed_data[day,:,0]) #Average percent_gl of all stocks for a particular day
			processed_data_dates[day,3] = np.std(processed_data[day,:,0]) #Standard dev of percent gl's
			processed_data_dates[day,4] = np.argmax(processed_data[day,:,0]) #Index of stock with greatest gain of the day
			processed_data_dates[day,5] = np.argmin(processed_data[day,:,0]) #Index of stock with greatest loss of the day
		
		#Calculate final metrics (Dependant on previous ones). Valid for even the 0-th date
		for stock_c in range(0, self.num_tickers):
			print('Second pass. Stock: ' + self.tickers[stock_c])
			for day in range(0, num_dates):
				percent_over_market = processed_data[day,stock_c,0] - processed_data_dates[day,2] #Percent out-performance of market
				if processed_data_dates[day,2] != 0:
					day_beta = processed_data[day,stock_c,0] / processed_data_dates[day,2] #Ratio out-performance of market
				else:
					day_beta = 0.0 #Avoid dividing by zero
				processed_data[day,stock_c,8:10] = [percent_over_market, day_beta]
			processed_data_stock[stock_c, 0] = np.average(processed_data[:,stock_c,0]) #Calculate all time average daily g/l for each stock
			processed_data_stock[stock_c, 1] = np.average(processed_data[:,stock_c,9]) #Calculate all time beta for stock
		#Save new processed data alongside old
		if add_to_existing and processed_data_exists:
			print('Saving new processed data alongside old')
			old_data = np.load(self.processed_data_path)
			combined_data = np.concatenate((old_data, processed_data), axis=0)
			np.save(self.processed_data_path, combined_data)
			combined_dates = old_dates.extend(dates)
		else:
			print('Saving processed data')
			combined_data = processed_data
			np.save(self.processed_data_path, combined_data)
			combined_dates = dates
		return combined_data, processed_data_stock, processed_data_dates, combined_dates
		
	#Fetches new data since the last data. Append new data to file. Processes data to calculate data metrics. Appends processed data metrics to file.
	#Return processed_data 
	def update_data(self, end_date):
		#Find the last date for which there is data saved
		with open(self.dates_path, mode='rb') as file:
			dates = pickle.load(file, encoding='bytes')
		old_data = np.load(self.processed_data_path)
		num_dates = len(dates)
		last_date = dt.datetime(1970,month=1,day=1)
		for dt_convert_c in range(0, num_dates):
			dt_date = convert_string_to_datetime(dates[dt_convert_c])
			if dt_date > last_date:
				last_date = dt_date
		start_dt = last_date + dt.timedelta(days=1)
		start_string = str(start_dt)
		start_string = start_string[0:10]
		#Get data
		new_data, new_dates = self.fetch_data(start_string, end_date)
		if len(new_dates) != 0:
			self.save_new_data(new_data, new_dates, True)
			numeric_data,_ = self.organize_data_from_vars(new_data, new_dates)
			processed_data,_, _, combined_dates = self.process_data(numeric_data, new_dates, True)
		else:
			print('Data is already up to date')
			processed_data = np.load(self.processed_data_path)
			new_dates = dates
		#Add new data to file, add the new dates to the dates file
		return processed_data, new_dates

#Input: datetime object, Output: datetime.date object
def convert_datetime_to_date(dt_):
	date_ = dt.date(dt_.year, dt_.month, dt_.day)
	return date_

#Convert datetime object into string
#Input: date object
#Output: String in mm/dd/yyyy format
def convert_date_to_string(input_date):
	day_str = str(input_date.year) + '-' + str(input_date.month) + '-' + str(input_date.day)
	return day_str

#Input: sting date 'yyyy-mm-dd' , Output: datetime.date object
def convert_string_to_date(input_string):
	output_dt = dt.datetime.strptime(input_string, '%Y-%m-%d')
	output = convert_datetime_to_date(output_dt)
	return output

def convert_datetime_to_string(input_dt):
	return convert_date_to_string(convert_datetime_to_date(input_dt))
	
#Input: string 'yyyy-mm-dd'
#Output: datetime date object
def convert_string_to_datetime(input_string):
	output = dt.datetime.strptime(input_string, '%Y-%m-%d')
	return output





