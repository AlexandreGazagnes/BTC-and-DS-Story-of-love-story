#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		Pre-processing
########################################################



# built in
from itertools import combinations
import pickle

# data management
import pandas as pd
import numpy as np
from stockstats import StockDataFrame

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



def create_df(filename) : 
	"""dataframe creation"""

	df = pd.read_csv(filename)
	df = StockDataFrame.retype(df)
	
	df.columns = [i.lower() for i in df.columns]

	ohlc = ["open", "close", "high", "low"] 
	for i in ohlc : 
		if i not in df.columns : 
			raise ValueError("df columns error : exepcted open, high, low, close")

	if "timestamp" not in df.columns : 
		raise ValueError("df columns error : exepcted timestamp")

	return df


def clean_data(df, filename=None) : 
	"""manage all data cleaning management : 
	drop na, duplicated values, nonsenses gaps"""

	# about na?

	print(df.isna().all())
	print(df.isna().any())
	print(df.isnull().all())
	print(df.isnull().any())

	# now if 2 days with have exact same values for open and close, is it wierd?
	# let's see this :

	df["same_close"] = df.close == df.close.shift()
	df["same_open"] = df.open == df.open.shift()
	df["duplicated"] = (df["same_close"] == True) & (df["same_open"] == True)

	print(df[df["duplicated"]])
	l = len(df[df["duplicated"]])
	print("number of dublicate sample in dfset = {} ({:.2f}%)"
		  .format(l, 100 * l/len(df)))

	# the best is yet to come ? How about ouuuuuut ... lieeeers !
	# first let's find if we have more than 50% of gap between 2 open, close...

	for feat in ["open", "close", "high", "low"]:
		gap_feat = "gap_{}".format(feat)
		df[gap_feat] = np.abs((df[feat] - df[feat].shift())/df[feat])
		df[gap_feat] = df[gap_feat] > 0.5
		print("number of {} with 50% gap :{}"
			  .format(feat, len(df[df[gap_feat]])))

	# drop useless new feat
	df.drop(['same_close', 'same_open', 'duplicated', 'gap_open',
       'gap_close', 'gap_high', 'gap_low'], axis=1, inplace=True)

	# save temp
	if filename : 
		df.to_csv(filename, index=False)

	return df


def add_features(df, feat_prices, ma_type, ma_period, filename=None): 
	"""proceed to feature engineering for df"""

	# drop useless values
	df.set_index("timestamp", inplace=True, drop=True)
	df = pd.DataFrame({"close": df["close"]})

	# create 12H_price > 24h_price etc etc
	for feat in feat_prices : 
		df[str(feat)] = df.close.shift(feat)

	for feat1, feat2 in combinations(df.columns, 2):
		df["_{}_{}".format(feat1, feat2)] = df[feat2] > df[feat1]

	# Drop na
	df.dropna(axis=0, how='any', inplace=True)

	# adding MMAs
	indicators = list()
	ma_period = pd.Series(ma_period).dropna().astype("int32")
	df = StockDataFrame.retype(df)

	for period in ma_period:

		# create close_X_sma
		ind = "close_{}_{}".format(period, ma_type)
		df.get(ind)
		df["_{}_{}".format("close", ind)] = df["close"] > df[ind]

		# add ind to indicators list
		indicators.append(ind)

		# add 12 vs ma, 24, vs ma etc etc
		for feat in feat_prices:
			df["_{}_{}".format(feat, ind)] = df[str(feat)] > df[ind]

	# add various ma vs eachother
	for feat1, feat2 in combinations(indicators, 2):
		df["_{}_{}".format(feat1, feat2)] = df[feat2] > df[feat1]

	# Drop na
	df.dropna(axis=0, how='any', inplace=True)

	# save temp
	if filename : 
		df.to_csv(filename, index=False)

	return df


def add_targets(df, target_prices, filename=None) :
	"""create one or more vectors targets"""

	# create target vectors
	for target in target_prices : 
		df["next_"+ str(target)] = df.close.shift(-target)

	target_list = list()
	for target in target_prices : 
		target_list.append("_close_next_"+ str(target))
		df["_close_next_"+ str(target)] = df["next_"+ str(target)] > df["close"]


	df.dropna(axis=0, how='any', inplace=True)

	return df, target_list


def drop_useless_feat(df, filename=None) : 
	""" keep from df only boll cretaded features"""

	# drop useless features
	col = [i for i in df.columns if i[0] != "_"]
	df.drop(col, axis=1, inplace=True)

	df.dropna(axis=0, how='any', inplace=True)

	# save temp
	if filename : 
		df.to_csv(filename, index=False)

	return df


def egalize_up_down(df, target) : 

	df_True = df[df[target] == True]
	df_False = df[df[target] == False]
	if len(df_True)>len(df_False) : 
		k = len(df_True) - len(df_False) 
		idx = np.random.choice(df_True.index, k, replace=False)
		df_True.drop(idx, axis=0, inplace=True)
		df = df_True.append(df_False)
	elif len(df_True)<len(df_False) : 
		k = len(df_False) - len(df_True)
		idx = np.random.choice(df_False.index, k, replace=False)
		df_False.drop(idx, axis=0, inplace=True)
		df = df_True.append(df_False)
	else : 
		pass

	return df

	