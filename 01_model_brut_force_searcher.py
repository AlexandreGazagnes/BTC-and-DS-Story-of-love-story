#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#####################################################################
#####################################################################
#	01_model_brut_force_searcher
#####################################################################
#####################################################################



########################################################
#		Description
########################################################






########################################################
#		Import
########################################################


# built in
from itertools import combinations
import pickle, datetime


# data management
import pandas as pd
import numpy as np
from stockstats import StockDataFrame

# visualization
import matplotlib.pyplot as plt
# %matplotlib
import seaborn as sns
sns.set()


# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score






########################################################
#		CONSTANTS
########################################################


DATA_FOLDER = './'
DATA_FILE = "BTC-USD_H.csv"

TEMP_FOLDER = "/media/alex/CL/"
TEMP_FILE = "temp.pk"

RESULT_FOLDER = "/media/alex/CL/"
RESULT_FILE = "result.pk"



########################################################
#		DataFrame Creation
########################################################


# dataframe creation

file = DATA_FOLDER+DATA_FILE
df = pd.read_csv(file)
df = StockDataFrame.retype(df)


# first santity check

print(df.head())
print(df.tail())
print(df.dtypes)
print(df.shape)
print(df.ndim)
print(df.describe())



########################################################
#		Datas Cleaning
########################################################


# about na?

print(df.isna().all())
print(df.isna().any())
print(df.isnull().all())
print(df.isnull().any())

# ok good !


# about null (0.0)?

print(df.describe().loc["min", :])


# ok not good for volume !
# let's count how many invalid values

for feat in df.columns.drop("time"):
	l = len(df[df[feat] == 0.0])
	print("number of null values for {} : {} ({:.2f}%)"
		  .format(feat, l, 100 * l/len(df)))


# very very small rate, we can delete df without impacting our work

df = df[df.volume != 0.0]


# now if 2 days with have exact same values for open and close, is it wierd?
# let's see this :

df["same_close"] = df.close == df.close.shift()
df["same_open"] = df.open == df.open.shift()
df["duplicated"] = (df["same_close"] == True) & (df["same_open"] == True)

print(df[df["duplicated"]])
l = len(df[df["duplicated"]])
print("number of dublicate sample in dfset = {} ({:.2f}%)"
	  .format(l, 100 * l/len(df)))


# very very small rate, we can delete df without impacting our work
# but before doin that, let's investigate this in depth

idx = df[df["duplicated"]].index
idx = [(i-1, i, i+1) for i in idx]
idx = np.array(idx).flatten()
print(df.loc[idx, :])


# ok this consistant and we can delete our invalid df

df = df[df["duplicated"] == False]
df.drop(["same_open", "same_close", "duplicated"], axis=1, inplace=True)


# the best is yet to come ? How about ouuuuuut ... lieeeers !
# first let's find if we have more than 50% of gap between 2 open, close...

for feat in ["open", "close", "high", "low"]:
	gap_feat = "gap_{}".format(feat)
	df[gap_feat] = np.abs((df[feat] - df[feat].shift())/df[feat])
	df[gap_feat] = df[gap_feat] > 0.5
	print("number of {} with 50% gap :{}"
		  .format(feat, len(df[df[gap_feat]])))

# ok not good at all

furious_gap = (df["gap_open"] == True) | (df["gap_close"] == True) | \
	(df["gap_low"] == True) | (df["gap_high"] == True)
df["furious_gap"] = furious_gap

# same as before let's investigate in depth

idx = df[df["furious_gap"]].index
idx = [(i-1, i, i+1) for i in idx]
idx = np.array(idx).flatten()
print(df.loc[idx, ["open", "high", "low", "close", "furious_gap"]])

# ok this consistant and we can delete our invalid df

df = df[df["furious_gap"] == False]
df.drop(["gap_open", "gap_close", "gap_low", "gap_high", "furious_gap"], axis=1, inplace=True)


# drop na?

print(df.isna().any())


# reindex

df.index = range(len(df))


# save temp

with open(TEMP_FOLDER+TEMP_FILE, "wb") as file : 
	pickle.dump(df, file)

del(df)



########################################################
#		META PARAMETRES AND LOOPER
########################################################


# initiate meta result handler

meta_results = list()
with open(RESULT_FOLDER+RESULT_FILE, "wb") as file : 
	pickle.dump(meta_results, file)


# define ma types

MA_TYPES = ("sma", "ema")


# define ma periods

# create first period tuples
ma_periods = [(i, i*j, i*k) for i in [25, 30, 50, 100]
			  for j in [1.25, 1.5, 2] for k in [3, 4, 5, 6, 8, 10]]
ma_periods.extend([(i, i*j, i*k) for i in [25, 30, 50, 100] for j in [3, 4, 5] for k in [6, 8, 10]])

# initiate dataframe and manage dtypes
ma_periods = pd.DataFrame(ma_periods, columns=["ma1", "ma2", "ma3"])
ma_periods["ma2"] = ma_periods.ma2.astype("int64")

# drop useless tuples
ma_periods["min_double"] = (
	ma_periods.ma2 * 2 <= ma_periods.ma3) & (ma_periods.ma2 * 4 >= ma_periods.ma3)
ma_periods = ma_periods[ma_periods["min_double"]]
ma_periods.drop(["min_double"], axis=1, inplace=True)
ma_periods.index = range(len(ma_periods))


# convert in numpy array 2d

# if you want all tuples
MA_PERIODS = ma_periods.values

# if you want x% of ma_periods 
rate = 1
mask = np.random.randint(0, len(ma_periods)+1, int(rate * len(ma_periods)))
ma_periods = ma_periods.loc[mask, :]
MA_PERIODS = ma_periods.values

# if you want to add specific values
specific_periods = pd.DataFrame([[30., 60., 180, np.nan], 
								 [100., 300., 600., np.nan], 
								 [100., 300., 1000., np.nan],
								 [100., 600., 1000., np.nan],
								 [100., 300., 600., 1000.]], 
								 columns=["ma1", "ma2", "ma3", "ma4"])

ma_periods = ma_periods.append(specific_periods)
MA_PERIODS = ma_periods.values

# # if you want just specific periods 
# MA_PERIODS=  specific_periods.values 


# build loop parser

MA_LIST = [(typ, per) for typ in MA_TYPES for per in MA_PERIODS]


# initiate loop
i=0
for (ma_type, ma_period) in MA_LIST:
	i +=1
	tot = len(MA_LIST) 
	print("""


		RUN {} sur {} -- {:.2f}%
	
		""".format(i, tot, 100 * i /tot))


	# intiate df 

	with open(TEMP_FOLDER+TEMP_FILE, "rb") as file : 
		df = pickle.load(file)



	########################################################
	#		Feature Engineering
	########################################################


	# focus an a specific period

	P = 12  #  for 12 H
	period = (P * 2) * 30 * 18  #  for 24h * 30 D * 18 months
	df = df[-period:]
	df.index = range(len(df))


	# drop useless values

	df.drop(["high", "low", "time", "open", "volume"], axis=1, inplace=True)


	# create 12H_price > 24h_price etc etc

	df["12"] = df.close.shift(P)
	df["24"] = df.close.shift(2 * P)
	df["36"] = df.close.shift(3 * P)
	df["48"] = df.close.shift(4 * P)
	df["60"] = df.close.shift(5 * P)
	df["72"] = df.close.shift(6 * P)

	for feat1, feat2 in combinations(df.columns, 2):
		df["_{}_{}".format(feat1, feat2)] = df[feat2] > df[feat1]


	# Drop na

	df.dropna(axis=0, how='any', inplace=True)


	# Santiy checks

	mask = ((df.index % P == 0) & (df.index < (P * 20)))
	print(df.loc[mask, ["close", "12", "24", "36", "next_D"]])

	print(df.head(5))
	print(df.tail(5))


	# adding MMAs

	indicators = list()
	ma_period = pd.Series(ma_period).dropna().astype("int32")

	for period in ma_period:

		# create close_X_sma
		ind = "close_{}_{}".format(period, ma_type)
		df.get(ind)
		df["_{}_{}".format("close", ind)] = df["close"] > df[ind]

		# add ind to indicators list
		indicators.append(ind)

		# add 12 vs ma, 24, vs ma etc etc
		for p in ["12", "24", "36", "48", "60", "72"]:
			df["_{}_{}".format(p, ind)] = df[p] > df[ind]

	# add various ma vs eachother
	for feat1, feat2 in combinations(indicators, 2):
		df["_{}_{}".format(feat1, feat2)] = df[feat2] > df[feat1]

	print(sorted(df.columns))


	# create target vectors
	df["next_24"] = df.close.shift(-2 * P)
	df["next_12"] = df.close.shift(-P)
	df["_close_next_24"] = df["next_24"] > df["close"]
	df["_close_next_12"] = df["next_12"] > df["close"]

	targets = ["_close_next_24", "_close_next_12"]

	df.dropna(axis=0, how='any', inplace=True)


	# drop useless features

	col = [i for i in df.columns if i[0] != "_"]
	print(col)
	df.drop(col, axis=1, inplace=True)
	print(df.columns)



	########################################################
	#		Not so brutal ML baseline
	########################################################


	# Seed

	np.random.seed(42)


	# For each target

	for target in targets:


		# create X,y

		y = df.loc[:, target]
		X = df.drop(targets, axis=1)


		# split test and train

		X_train, X_test, y_train, y_test \
			= train_test_split(X, y, shuffle=True, stratify=y, random_state=42)

		print([k.shape for k in [X_train, X_test, y_train, y_test]])


		# model and params list

		model_list = [RandomForestClassifier, ]
		params_list = [{'n_estimators': [300],  #  [100, 300, 500], 	# [100, 300, 500, 700, 1000],
						# depriciated : [int(i) for i in np.logspace(2, 3, 5)],
						'bootstrap': [True],
						'oob_score':  [True],
						'warm_start':  [True]}]


		# GridSearchCV

		for Model, params in zip(model_list, params_list):


			# init model and grid

			model = Model(random_state=42, n_jobs=3, verbose=0)
			grid = GridSearchCV(model, params, cv=5, verbose=2, scoring="accuracy")


			# fit and pred

			grid.fit(X_train, y_train)
			y_pred = grid.predict(X_test)
			y_prob = grid.predict_proba(X_test)[:, 0]


			# save results to handle it in post prod

			results = pd.DataFrame({"prob": y_prob, "pred": y_pred, "test": y_test})
			results["good_pred"] = y_pred == y_test


			# print brut results

			print("\nparams : {}\n".format(grid.best_params_))
			print("score avec {} sur la target {}: {:.4f}"
				  .format(Model.__name__, target, grid.score(X_test, y_test)))


			# show base line

			print("mais % de hausse sur l'echantillon : {:.4f}\n\n"
				  .format(len(y[y == True])/len(y)))


			# record for meta_results

			ma_period = list(ma_period)

			if len(ma_period) == 3:
				ma_period.append(np.nan)

			with open(RESULT_FOLDER+RESULT_FILE, "rb") as file : 
				meta_results = pickle.load(file)

			meta_results.append([grid.score(X_test, y_test), grid.best_score_,
								 target, ma_type, *ma_period,
								 grid.best_params_, results, "str of grid"])

			with open(RESULT_FOLDER+RESULT_FILE, "wb") as file : 
				pickle.dump(meta_results, file)



###############################################################
#  		Analysing results
###############################################################


# load results 

with open(RESULT_FOLDER+RESULT_FILE, "rb") as file : 
	meta_results = pickle.load(file)

# transform in DataFrame

meta_results.sort(key=lambda x: x[0], reverse=True)
meta_results = pd.DataFrame(meta_results)
meta_results.columns = ["score_1", "score_2", "target", "ma", "p1","p2","p3",
						"p4", "params", "results", "grid"]


with open(RESULT_FOLDER+RESULT_FILE, "wb") as file : 
	pickle.dump(meta_results, file)


# first we  are going to trade of with prob rate (predict proba)

meta_prob_results = pd.DataFrame(index=range(51, 100))
meta_prob_quants = pd.DataFrame(index=range(51, 100))

for i in meta_results.index : 
	results = meta_results.loc[i, "results"]
	label = "{}|{}|{}|{}|{}".format(*list(meta_results.iloc[i, 3:8]))

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



meta_prob_results.iloc[:, :20].plot()
plt.show()	


meta_prob_quants.iloc[:, :20].plot()
plt.show()	
 
efficiancy = meta_prob_results.iloc[:, 0] * meta_prob_quants.iloc[:, 0]
efficiancy.plot()

performance = meta_prob_results.iloc[:, 0] / meta_prob_quants.iloc[:, 0]
performance.plot()
plt.show()



# second we are going to analyse trade of with prob rate (predict proba) for 
# each good or false value


results = meta_results.loc[0, "results"]

true_results = list()
false_results = list()
proba_quants = list()
i_j = [(50-k, 50+k) for k in range(1, 50)]

for i, j in i_j:
	mask = (results["prob"] > (i/100)) & (results["prob"] < (j/100))
	mask = ~mask
	sub_result = results[mask]
	true = len(sub_result[sub_result["good_pred"] == True])
	false = len(sub_result[sub_result["good_pred"] == False])

	true_results.append(true/len(sub_result))
	false_results.append(false/len(sub_result))
	proba_quants.append(len(sub_result)/len(results))

true_false_df = pd.DataFrame({"true" : true_results, "false" : false_results, "quants" : proba_quants},
				index= range(51,100))

true_false_df["true_"] = true_false_df["true"] * true_false_df["quants"]
true_false_df["false_"] = true_false_df["false"] * true_false_df["quants"]

true_false_df.drop(["quants", "true", "false"], axis=1)

true_false_df.plot()
plt.show()



true_false_df["true_"] = true_false_df["true"] /true_false_df["quants"]
true_false_df["false_"] = true_false_df["false"] / true_false_df["quants"]

true_false_df.drop(["quants", "true", "false"], axis=1)

true_false_df.plot()
plt.show()



# ok so, when proba rise, accuracy rsie, somethnhg linear
# but, acuuracy has to be less or geal than 1.0
# so we mih excpet somting loaritmik
# we can se that globaly whe have acuracy doing spectacluar thin between 90%
# and 99% of prob 
# somtinge rise spectacular, sometimes fall spectacular, why ???
# beccause wheb prob exigence rise, % of results stuides / total results falls
# trending to 0 ! so results so "effect of size" is more and more important, and 
# the more we have a small dataset the more wen could see spectaular data
# so a good trade of is between 80 and 90%
# very good could be 90% 
# but as we might be carefull œwe will chose somtehing betwtewnn 85 and 90% 

# very intessing is taht for each exeprience, corelation betwen predict proba 
# and accuracy for  51 to 90% follow more a less the same trend
# this is very good : for any exp could excpect a prety confident best  result betwenn 
# 80 and 90% (we drop more than 90, could be better but to cahotic to be studied)


# let s now deep in hard job : 
# what about ma1, 2, 3 value and result???

# lets go for a new data set ???

feat_impact = list();
features = ['target', 'ma', 'p1', 'p2', 'p3', 'p4']; 
for feat in features : 
	data = pd.concat([meta_results["score_1"], meta_results[feat]], axis=1)
	data = data.groupby(feat).describe().loc[:,"score_1"].round(2)
	feat_impact.append(data)
	print(data)


feat_impact = list();
features = ['p1', 'p2', 'p3']; 
for feat in features : 
	data = pd.concat([meta_results["score_1"], meta_results[feat]], axis=1)
	plt.plot(meta_results[feat], meta_results["score_1"])
	plt.show()


# we can see that best is prev to 24h with SMA 100, 300, 600, 1000 ecte ct
# lets do machine learning for this ! 
# it is up to you :) 
