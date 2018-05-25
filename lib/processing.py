#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		Processing
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


