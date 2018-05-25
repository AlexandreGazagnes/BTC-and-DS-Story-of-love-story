#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		Functions
########################################################


def manage_imports() : 
	""" manage all imports in once"""

	global combinations, pickle
	global pd, np, StockDataFrame
	global plt, sns
	global train_test_split, GridSearchCV, RandomForestClassifier, accuracy_score


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



manage_imports()



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


def create_meta_results(col) : 
	""" initiate a empty result handler with columns pre defined"""
	
	meta_results = pd.DataFrame(columns=col)
	
	return meta_results


def create_X_y(df, target, targets) : 
	y = df.loc[:, target]
	X = df.drop(targets, axis=1)

	return X, y


def split_X_y(X,y,test_size=0.25, shuffle=True, stratify=None) : 
	X_train, X_test, y_train, y_test \
		= train_test_split(X, y, shuffle=shuffle, stratify=stratify, 
				test_size = test_size, train_size = 1 - test_size, 
				random_state=42)

	print([k.shape for k in [X_train, X_test, y_train, y_test]])

	return X_train, X_test, y_train, y_test 


def grid_init(Model, params, CV=5, njobs=6, verb=2, scor="accuracy") : 
	""" init"""

	model = Model(random_state=42, n_jobs=njobs, verbose=0)
	grid = GridSearchCV(model, params, cv=CV, verbose=verb, scoring=scor)

	return grid


def grid_fit(grid, X_train, y_train) : 
	""" fit"""
	
	grid.fit(X_train, y_train)
	return grid


def grid_results(grid, X_test, y_test) : 
	""" build a specific dataframe with pred, proba, test"""
	y_pred = grid.predict(X_test)
	y_prob = grid.predict_proba(X_test)[:, 0]
	results = pd.DataFrame({"prob": y_prob, "pred": y_pred, "test": y_test})
	results["good_pred"] = y_pred == y_test

	return results


def grid_score(results) : 
	"""compute basics accruacy score"""
	return len(results.loc[results["good_pred"] == True, :]) / len(results)


def update_results(new_result, columns, filename) : 
	"""update bianiry file of meta_results """

	if len(new_result) != len(columns) : 
		raise ValueError("meta_results : len features in new_results != len meta_results.columnsumns")

	meta_results = load_bin_results(filename)
	new_result = pd.Series(new_result, index = columns)
	meta_results = meta_results.append(new_result, ignore_index=True)
	save_bin_results(meta_results, filename)


def save_bin_results(meta_results, filename) : 
	""" save meta results in binary object file"""
	with open(filename, "wb") as file : 
		pickle.dump(meta_results, file)


def load_bin_results(filename) : 
	""" load meta results from binary format"""
	with open(filename, "rb") as file : 
		df = pickle.load(file)
	return df


def readablize_results(meta_results, columns) : 
	return meta_results.drop(columns, axis=1)

def delete_temps():
	pass

def create_meta_probs(meta_results, threshold=None, param=None) : 

	if not threshold : 

		meta_prob_results = pd.DataFrame(index=range(51, 100))
		meta_prob_quants = pd.DataFrame(index=range(51, 100))

		for i in meta_results.index : 
			results = meta_results.loc[i, "results"]
			
			label = meta_results.loc[i, param] if param else meta_results.index[i]

			proba_results = list()
			proba_quants = list()

			i_j = [(50-k, 50+k) for k in range(1, 50)]

			for i, j in i_j:
				mask = (results["prob"] > (i/100)) & (results["prob"] < (j/100))
				mask = ~mask
				sub_result = results[mask]
				k = len(sub_result[sub_result["good_pred"] == True])

				proba_results.append(k/len(sub_result))
				proba_quants.append(len(sub_result)/len(results))

			meta_prob_results[label] = proba_results
			meta_prob_quants[label] = proba_quants

		return meta_prob_results, meta_prob_quants

	else  : 

		proba_results = list()
		proba_quants = list()

		for i in meta_results.index : 

			results = meta_results.loc[i, "results"]
			mask = (results["prob"] > (1-threshold)) & (results["prob"] < threshold)
			mask = ~mask
			sub_result = results[mask]
			k = len(sub_result[sub_result["good_pred"] == True])
			proba_results.append(k/len(sub_result))
			proba_quants.append(len(sub_result)/len(results))

		n = int(100 * threshold)
		meta_results["prob{}_score".format(n)] = proba_results
		meta_results["prob{}_quant".format(n)] = proba_quants


def split_meta_results(meta_results, target) : 
	pass


