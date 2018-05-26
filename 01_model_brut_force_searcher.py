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
#		Imports and configs
########################################################


# lib
from lib.filepaths import *
from lib.preprocessing import *
from lib.processing import *
from lib.results import *



########################################################
#		Main
########################################################



# File paths
####################################

DATA_FILE, CLEANED_DF_FILE, \
	PREPARED_DF_FILE, RESULT_BIN_FILE, RESULT_DF_FILE = build_file_paths()



# Constants and Meta params
#####################################

# meta results format
META_RESULTS_COL = ["score", "target", "results", "grid"]

# features prices
P, NUMB = 12, 6
FEAT_PRICES = [(i+1) * P for i in range(NUMB)]

# features MAs
MA_TYPE ["ema", "sma"]

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
MA_PERIOD = ma_periods.values

# if you want x% of ma_periods 
rate = 1
mask = np.random.randint(0, len(ma_periods)+1, int(rate * len(ma_periods)))
ma_periods = ma_periods.loc[mask, :]
MA_PERIOD = ma_periods.values

# if you want to add specific values
specific_periods = pd.DataFrame([[30., 60., 180, np.nan], 
								 [100., 300., 600., np.nan], 
								 [100., 300., 1000., np.nan],
								 [100., 600., 1000., np.nan],
								 [100., 300., 600., 1000.]], 
								 columns=["ma1", "ma2", "ma3", "ma4"])

ma_periods = ma_periods.append(specific_periods)
MA_PERIOD = ma_periods.values

# # if you want just specific periods 
# MA_PERIODS=  specific_periods.values 

# build loop parser
MA_LIST = [(typ, per) for typ in MA_TYPE for per in MA_PERIOD]


# target prices
TARGET_PRICES = [P, 2*P]

# ML params
MODEL_LIST = [RandomForestClassifier]
MODEL_PARAMS = [{'n_estimators': [300, 500, 700, 1000], 'bootstrap': [True, False], 
							'oob_score':  [True,False], 'warm_start':  [True,False]}]
TEST_SIZE  = [0.3]
CV = 10

# results params	
DROP_RESULTS_COLS = ["results", "grid"]



# Pre-processing
#####################################

# initiate and save meta_results 
meta_results = create_meta_results(META_RESULTS_COL)
save_bin_results(meta_results, RESULT_BIN_FILE)


for ma_type, ma_period in MA_LIST : 
	
	# intiate and clean dataframe
	df = create_df(DATA_FILE)
	df = clean_data(df, CLEANED_DF_FILE)

	# feat ingeenirng and targets
	df = add_features(df, FEAT_PRICES, ma_type, ma_period, PREPARED_DF_FILE)
	df, targets = add_targets(df, TARGET_PRICES, PREPARED_DF_FILE)
	df = drop_useless_feat(df, PREPARED_DF_FILE)



	# Processing
	#####################################

	for target in targets : 

		# prepare X, y test and train
		X,y = create_X_y(df, target, targets)
		X_train, X_test, y_train, y_test \
			= split_X_y(X,y,test_size=test_size, shuffle=True, stratify=y) 

		for Model, model_params in zip(MODEL_LIST, MODEL_PARAMS) : 

			# init and fit model
			grid = grid_init(Model, model_params, CV=CV)
			grid = grid_fit(grid, X_train, y_train)

			# manage results
			results = grid_results(grid, X_test,y_test)
			score = grid_score(results)

			# print brut results
			print("\nparams : {}".format(grid.best_params_))
			print("score avec {} sur la target {}: {:.4f}\n"
			  .format(Model.__name__, target, score))

			# # show base line
			# print("mais % de hausse sur l'echantillon : {:.4f}\n\n"
			# 	  .format(len(y[y == True])/len(y)))

			# update meta_results
			new_result = [score, target, results, grid.best_params_]
			update_results(new_result, META_RESULTS_COL, RESULT_BIN_FILE)



# Analysing Results
######################################

# gather meta_results bin and csv
meta_results = load_bin_results(RESULT_BIN_FILE)	
meta_results_ = readablize_results(meta_results, DROP_RESULTS_COLS)
meta_results_.to_csv(RESULT_DF_FILE, index= False)

# handle predict proba impact
create_meta_probs(meta_results, threshold=0.85)
meta_results_ = readablize_results(meta_results, DROP_RESULTS_COLS)
meta_probs, meta_quants = create_meta_probs(meta_results, param="target", graph=True)
