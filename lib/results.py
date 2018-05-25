#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		Results
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



def create_meta_results(col) : 
	""" initiate a empty result handler with columns pre defined"""
	
	meta_results = pd.DataFrame(columns=col)
	
	return meta_results


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


def create_meta_probs(meta_results, threshold=None, graph=None,param=None) : 

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

				if len(sub_result) == 0  : 
					k = 0
				else : 
					k = k/len(sub_result)					

				proba_results.append(k)
				proba_quants.append(len(sub_result)/len(results))

			meta_prob_results[label] = proba_results
			meta_prob_quants[label] = proba_quants

		if graph==True : 
			# plot
			fig, (ax0, ax1) = plt.subplots(1, 2)

			meta_prob_results.plot(ax=ax0)
			ax0.set_xlabel("predict_proba")
			ax0.set_ylabel("score")
			ax0.legend()
			ax0.set_title("score evolution vs predict proba")	

			meta_prob_quants.plot(ax=ax1)
			ax1.set_xlabel("predict_proba")
			ax1.set_ylabel("% of previsions selected")
			ax1.legend()
			ax1.set_title("% of previsions selected vs predict proba")	
			plt.suptitle("results of proba and quants")

			plt.show()

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

			if len(sub_result) == 0  : 
				k = 0
			else : 
				k = k/len(sub_result)					

			proba_results.append(k)
			proba_quants.append(len(sub_result)/len(results))

		n = int(100 * threshold)
		meta_results["prob{}_score".format(n)] = proba_results
		meta_results["prob{}_quant".format(n)] = proba_quants

